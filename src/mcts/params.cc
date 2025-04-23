/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header) ...
*/

#include "mcts/params.h" // Adjusted include path

#include <algorithm>
#include <cctype>
#include <cmath>

#include "neural/shared_params.h" // Include shared params
#include "utils/exception.h"
#include "utils/string.h"

#if __has_include("params_override.h")
#include "params_override.h"
#endif

#ifndef DEFAULT_MAX_PREFETCH
#define DEFAULT_MAX_PREFETCH 32
#endif
#ifndef DEFAULT_TASK_WORKERS
// Use a default appropriate for Ergodice-like forks if different
#define DEFAULT_TASK_WORKERS -1 // Default to heuristic based on CPU/GPU
#endif

namespace lczero { // No classic namespace

namespace {
FillEmptyHistory EncodeHistoryFill(std::string history_fill) {
  if (history_fill == "fen_only") return FillEmptyHistory::FEN_ONLY;
  if (history_fill == "always") return FillEmptyHistory::ALWAYS;
  assert(history_fill == "no");
  return FillEmptyHistory::NO;
}

float GetContempt(std::string name, std::string contempt_str,
                  float uci_rating_adv) {
  float contempt = uci_rating_adv;
  for (auto& entry : StrSplit(contempt_str, ",")) {
    if (entry.length() == 0) continue;
    auto parts = StrSplit(entry, "=");
    if (parts.size() == 1) {
      try { contempt = std::stof(parts[0]); } catch (std::exception& e) { throw Exception("Invalid default contempt: " + entry); }
    } else if (parts.size() == 2) {
      if (std::search(name.begin(), name.end(), parts[0].begin(), parts[0].end(),
                      [](unsigned char a, unsigned char b) { return std::tolower(a) == std::tolower(b); }) != name.end()) {
        try { contempt = std::stof(parts[1]); } catch (std::exception& e) { throw Exception("Invalid contempt entry: " + entry); }
        break;
      }
    } else {
      throw Exception("Invalid contempt entry:" + entry);
    }
  }
  return contempt;
}

SearchParams::WDLRescaleParams AccurateWDLRescaleParams(
    float contempt, float draw_rate_target, float draw_rate_reference,
    float book_exit_bias, float contempt_max, float contempt_attenuation) {
  if (draw_rate_target > 0.0f && draw_rate_target < 0.001f) draw_rate_target = 0.001f;
  float scale_reference = 1.0f / std::log((1.0f + draw_rate_reference) / (1.0f - draw_rate_reference));
  float scale_target = (draw_rate_target == 0 ? scale_reference : 1.0f / std::log((1.0f + draw_rate_target) / (1.0f - draw_rate_target)));
  float ratio = scale_target / scale_reference;
  float diff = scale_target / (scale_reference * scale_reference) /
               (1.0f / std::pow(std::cosh(0.5f * (1 - book_exit_bias) / scale_target), 2) +
                1.0f / std::pow(std::cosh(0.5f * (1 + book_exit_bias) / scale_target), 2)) *
               std::log(10) / 200 * std::clamp(contempt, -contempt_max, contempt_max) * contempt_attenuation;
  return SearchParams::WDLRescaleParams(ratio, diff);
}

float ConvertRegularToGamePairElo(float elo_regular) {
  const float transition_sharpness = 250.0f;
  const float transition_midpoint = 2737.0f;
  return elo_regular +
         0.5f * transition_sharpness *
             std::log(1.0f + std::exp((transition_midpoint - elo_regular) / transition_sharpness));
}

SearchParams::WDLRescaleParams SimplifiedWDLRescaleParams(
    float contempt, float draw_rate_reference, float elo_active,
    float contempt_max, float contempt_attenuation) {
  const float scale_zero = 15.0f; const float elo_slope = 425.0f; const float offset = 6.75f;
  float scale_reference = 1.0f / std::log((1.0f + draw_rate_reference) / (1.0f - draw_rate_reference));
  float elo_opp = elo_active - std::clamp(contempt, -contempt_max, contempt_max);
  elo_active = ConvertRegularToGamePairElo(elo_active);
  elo_opp = ConvertRegularToGamePairElo(elo_opp);
  float scale_active = 1.0f / (1.0f / scale_zero + std::exp(elo_active / elo_slope - offset));
  float scale_opp = 1.0f / (1.0f / scale_zero + std::exp(elo_opp / elo_slope - offset));
  float scale_target = std::sqrt((scale_active * scale_active + scale_opp * scale_opp) / 2.0f);
  float ratio = scale_target / scale_reference;
  float mu_active = -std::log(10) / 200 * scale_zero * elo_slope * std::log(1.0f + std::exp(-elo_active / elo_slope + offset) / scale_zero);
  float mu_opp = -std::log(10) / 200 * scale_zero * elo_slope * std::log(1.0f + std::exp(-elo_opp / elo_slope + offset) / scale_zero);
  float diff = 1.0f / (scale_reference * scale_reference) * (mu_active - mu_opp) * contempt_attenuation;
  return SearchParams::WDLRescaleParams(ratio, diff);
}
}  // namespace

// --- Define OptionIds ---
const OptionId SearchParams::kMiniBatchSizeId{ /* ... */ };
const OptionId SearchParams::kMaxPrefetchBatchId{ /* ... */ };
const OptionId SearchParams::kCpuctId{ /* ... */ };
const OptionId SearchParams::kCpuctAtRootId{ /* ... */ };
const OptionId SearchParams::kCpuctExponentId{ /* ... */ };
const OptionId SearchParams::kCpuctExponentAtRootId{ /* ... */ };
const OptionId SearchParams::kCpuctBaseId{ /* ... */ };
const OptionId SearchParams::kCpuctBaseAtRootId{ /* ... */ };
const OptionId SearchParams::kCpuctFactorId{ /* ... */ };
const OptionId SearchParams::kCpuctFactorAtRootId{ /* ... */ };
const OptionId SearchParams::kRootHasOwnCpuctParamsId{ /* ... */ };
const OptionId SearchParams::kTwoFoldDrawsId{ /* ... */ };
const OptionId SearchParams::kTemperatureId{ /* ... */ };
const OptionId SearchParams::kTempDecayMovesId{ /* ... */ };
const OptionId SearchParams::kTempDecayDelayMovesId{ /* ... */ };
const OptionId SearchParams::kTemperatureCutoffMoveId{ /* ... */ };
const OptionId SearchParams::kTemperatureEndgameId{ /* ... */ };
const OptionId SearchParams::kTemperatureWinpctCutoffId{ /* ... */ };
const OptionId SearchParams::kTemperatureVisitOffsetId{ /* ... */ };
const OptionId SearchParams::kNoiseEpsilonId{ /* ... */ };
const OptionId SearchParams::kNoiseAlphaId{ /* ... */ };
const OptionId SearchParams::kVerboseStatsId{ /* ... */ };
const OptionId SearchParams::kLogLiveStatsId{ /* ... */ };
const OptionId SearchParams::kFpuStrategyId{ /* ... */ };
const OptionId SearchParams::kFpuValueId{ /* ... */ };
const OptionId SearchParams::kFpuStrategyAtRootId{ /* ... */ };
const OptionId SearchParams::kFpuValueAtRootId{ /* ... */ };
const OptionId SearchParams::kCacheHistoryLengthId{ /* ... */ };
// kPolicySoftmaxTempId likely uses SharedBackendParams::kPolicySoftmaxTemp
const OptionId SearchParams::kMaxCollisionEventsId{ /* ... */ };
const OptionId SearchParams::kMaxCollisionVisitsId{ /* ... */ };
const OptionId SearchParams::kOutOfOrderEvalId{ /* ... */ };
const OptionId SearchParams::kStickyEndgamesId{ /* ... */ };
const OptionId SearchParams::kSyzygyFastPlayId{ /* ... */ };
const OptionId SearchParams::kMultiPvId{ /* ... */ };
const OptionId SearchParams::kPerPvCountersId{ /* ... */ };
const OptionId SearchParams::kScoreTypeId{ /* ... */ };
// kHistoryFillId likely uses SharedBackendParams::kHistoryFill
const OptionId SearchParams::kMovesLeftMaxEffectId{ /* ... */ };
const OptionId SearchParams::kMovesLeftThresholdId{ /* ... */ };
const OptionId SearchParams::kMovesLeftConstantFactorId{ /* ... */ };
const OptionId SearchParams::kMovesLeftScaledFactorId{ /* ... */ };
const OptionId SearchParams::kMovesLeftQuadraticFactorId{ /* ... */ };
const OptionId SearchParams::kMovesLeftSlopeId{ /* ... */ };
const OptionId SearchParams::kDisplayCacheUsageId{ /* ... */ };
const OptionId SearchParams::kMaxConcurrentSearchersId{ /* ... */ };
const OptionId SearchParams::kDrawScoreId{ /* ... */ };
const OptionId SearchParams::kContemptModeId{ /* ... */ };
const OptionId SearchParams::kContemptId{ /* ... */ };
const OptionId SearchParams::kContemptMaxValueId{ /* ... */ };
const OptionId SearchParams::kWDLCalibrationEloId{ /* ... */ };
const OptionId SearchParams::kWDLContemptAttenuationId{ /* ... */ };
const OptionId SearchParams::kWDLMaxSId{ /* ... */ };
const OptionId SearchParams::kWDLEvalObjectivityId{ /* ... */ };
const OptionId SearchParams::kWDLDrawRateTargetId{ /* ... */ };
const OptionId SearchParams::kWDLDrawRateReferenceId{ /* ... */ };
const OptionId SearchParams::kWDLBookExitBiasId{ /* ... */ };
const OptionId SearchParams::kMaxOutOfOrderEvalsFactorId{ /* ... */ };
const OptionId SearchParams::kNpsLimitId{ /* ... */ };
const OptionId SearchParams::kSolidTreeThresholdId{ /* ... */ };
const OptionId SearchParams::kTaskWorkersPerSearchWorkerId{ /* ... */ };
const OptionId SearchParams::kMinimumWorkSizeForProcessingId{ /* ... */ };
const OptionId SearchParams::kMinimumWorkSizeForPickingId{ /* ... */ };
const OptionId SearchParams::kMinimumRemainingWorkSizeForPickingId{ /* ... */ };
const OptionId SearchParams::kMinimumWorkPerTaskForProcessingId{ /* ... */ };
const OptionId SearchParams::kIdlingMinimumWorkId{ /* ... */ };
const OptionId SearchParams::kThreadIdlingThresholdId{ /* ... */ };
const OptionId SearchParams::kMaxCollisionVisitsScalingStartId{ /* ... */ };
const OptionId SearchParams::kMaxCollisionVisitsScalingEndId{ /* ... */ };
const OptionId SearchParams::kMaxCollisionVisitsScalingPowerId{ /* ... */ };
const OptionId SearchParams::kUCIOpponentId{ /* ... */ };
const OptionId SearchParams::kUCIRatingAdvId{ /* ... */ };
const OptionId SearchParams::kSearchSpinBackoffId{ /* ... */ };
// --- Add Definitions for any other OptionIds assumed from Ergodice if needed ---
// const OptionId SearchParams::kUseVarianceScalingId { ... };

// --- Root Beam Search ADDED ---
const OptionId SearchParams::kRootBeamMinWidthId{
    "root-beam-min-width", "RootBeamMinWidth",
    "Minimum beam width when using dynamic width (based on score gap). Set to 0 or >= MaxWidth to disable dynamic width."};
const OptionId SearchParams::kRootBeamMaxWidthId{
    "root-beam-max-width", "RootBeamMaxWidth",
    "Maximum beam width (or fixed width if MinWidth=0/disabled). 0 disables beam."};
const OptionId SearchParams::kRootBeamUpdateThresholdId{
    "root-beam-update-threshold", "RootBeamUpdateThreshold",
    "Number of root visits after which the root beam is calculated and activated."};
const OptionId SearchParams::kRootBeamUpdateIntervalFactorId{
    "root-beam-update-interval-factor", "RootBeamUpdateIntervalFactor",
    "Geometric factor to increase update interval (>1.0 enables geometric). 1.0 means fixed interval."};
// --- END Root Beam Search ADDED ---


void SearchParams::Populate(OptionsParser* options) {
  // Add options using options->Add<>(...)
  options->Add<IntOption>(kMiniBatchSizeId, 0, 1024) = 0;
  options->Add<IntOption>(kMaxPrefetchBatchId, 0, 1024) = DEFAULT_MAX_PREFETCH;
  options->Add<FloatOption>(kCpuctId, 0.0f, 100.0f) = 1.745f; // Use appropriate default
  options->Add<FloatOption>(kCpuctAtRootId, 0.0f, 100.0f) = 1.745f; // Use appropriate default
  options->Add<FloatOption>(kCpuctExponentId, 0.0f, 1.0f) = 0.5f; // Use appropriate default
  options->Add<FloatOption>(kCpuctExponentAtRootId, 0.0f, 1.0f) = 0.5f; // Use appropriate default
  options->Add<FloatOption>(kCpuctBaseId, 1.0f, 1000000000.0f) = 38739.0f; // Use appropriate default
  options->Add<FloatOption>(kCpuctBaseAtRootId, 1.0f, 1000000000.0f) = 38739.0f; // Use appropriate default
  options->Add<FloatOption>(kCpuctFactorId, 0.0f, 1000.0f) = 3.894f; // Use appropriate default
  options->Add<FloatOption>(kCpuctFactorAtRootId, 0.0f, 1000.0f) = 3.894f; // Use appropriate default
  options->Add<BoolOption>(kRootHasOwnCpuctParamsId) = false;
  options->Add<BoolOption>(kTwoFoldDrawsId) = true;
  options->Add<FloatOption>(kTemperatureId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kTempDecayMovesId, 0, 640) = 0;
  options->Add<IntOption>(kTempDecayDelayMovesId, 0, 100) = 0;
  options->Add<IntOption>(kTemperatureCutoffMoveId, 0, 1000) = 0;
  options->Add<FloatOption>(kTemperatureEndgameId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kTemperatureWinpctCutoffId, 0.0f, 100.0f) = 100.0f;
  options->Add<FloatOption>(kTemperatureVisitOffsetId, -1000.0f, 1000.0f) = 0.0f;
  options->Add<FloatOption>(kNoiseEpsilonId, 0.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kNoiseAlphaId, 0.0f, 10000000.0f) = 0.3f;
  options->Add<BoolOption>(kVerboseStatsId) = false;
  options->Add<BoolOption>(kLogLiveStatsId) = false;
  std::vector<std::string> fpu_strategy = {"reduction", "absolute"};
  options->Add<ChoiceOption>(kFpuStrategyId, fpu_strategy) = "reduction";
  options->Add<FloatOption>(kFpuValueId, -100.0f, 100.0f) = 0.330f; // Use appropriate default
  fpu_strategy.push_back("same");
  options->Add<ChoiceOption>(kFpuStrategyAtRootId, fpu_strategy) = "same";
  options->Add<FloatOption>(kFpuValueAtRootId, -100.0f, 100.0f) = 1.0f;
  options->Add<IntOption>(kCacheHistoryLengthId, 0, 7) = 0;
  // kPolicySoftmaxTempId likely populated by SharedBackendParams::Populate
  options->Add<IntOption>(kMaxCollisionEventsId, 1, 65536) = 917;
  options->Add<IntOption>(kMaxCollisionVisitsId, 1, 100000000) = 80000;
  options->Add<IntOption>(kMaxCollisionVisitsScalingStartId, 1, 100000) = 28;
  options->Add<IntOption>(kMaxCollisionVisitsScalingEndId, 0, 100000000) = 145000;
  options->Add<FloatOption>(kMaxCollisionVisitsScalingPowerId, 0.01, 100) = 1.25;
  options->Add<BoolOption>(kOutOfOrderEvalId) = true;
  options->Add<FloatOption>(kMaxOutOfOrderEvalsFactorId, 0.0f, 100.0f) = 2.4f;
  options->Add<BoolOption>(kStickyEndgamesId) = true;
  options->Add<BoolOption>(kSyzygyFastPlayId) = false;
  options->Add<IntOption>(kMultiPvId, 1, 500) = 1;
  options->Add<BoolOption>(kPerPvCountersId) = false;
  std::vector<std::string> score_type = {"centipawn", "centipawn_with_drawscore", "centipawn_2019", "centipawn_2018", "win_percentage", "Q", "W-L", "WDL_mu"};
  options->Add<ChoiceOption>(kScoreTypeId, score_type) = "WDL_mu";
  // kHistoryFillId likely populated by SharedBackendParams::Populate
  options->Add<FloatOption>(kMovesLeftMaxEffectId, 0.0f, 1.0f) = 0.0345f;
  options->Add<FloatOption>(kMovesLeftThresholdId, 0.0f, 1.0f) = 0.8f;
  options->Add<FloatOption>(kMovesLeftSlopeId, 0.0f, 1.0f) = 0.0027f;
  options->Add<FloatOption>(kMovesLeftConstantFactorId, -1.0f, 1.0f) = 0.0f;
  options->Add<FloatOption>(kMovesLeftScaledFactorId, -2.0f, 2.0f) = 1.6521f;
  options->Add<FloatOption>(kMovesLeftQuadraticFactorId, -1.0f, 1.0f) = -0.6521f;
  options->Add<BoolOption>(kDisplayCacheUsageId) = false;
  options->Add<IntOption>(kMaxConcurrentSearchersId, 0, 128) = 1;
  options->Add<FloatOption>(kDrawScoreId, -1.0f, 1.0f) = 0.0f;
  std::vector<std::string> mode = {"play", "white_side_analysis", "black_side_analysis", "disable"};
  options->Add<ChoiceOption>(kContemptModeId, mode) = "play";
  options->Add<StringOption>(kContemptId) = "";
  options->Add<FloatOption>(kContemptMaxValueId, 0, 10000.0f) = 420.0f;
  options->Add<FloatOption>(kWDLCalibrationEloId, 0, 10000.0f) = 0.0f;
  options->Add<FloatOption>(kWDLContemptAttenuationId, -10.0f, 10.0f) = 1.0f;
  options->Add<FloatOption>(kWDLMaxSId, 0.0f, 10.0f) = 1.4f;
  options->Add<FloatOption>(kWDLEvalObjectivityId, 0.0f, 1.0f) = 1.0f;
  options->Add<FloatOption>(kWDLDrawRateTargetId, 0.0f, 0.999f) = 0.0f;
  options->Add<FloatOption>(kWDLDrawRateReferenceId, 0.001f, 0.999f) = 0.5f;
  options->Add<FloatOption>(kWDLBookExitBiasId, -2.0f, 2.0f) = 0.65f;
  options->Add<FloatOption>(kNpsLimitId, 0.0f, 1e6f) = 0.0f;
  options->Add<IntOption>(kSolidTreeThresholdId, 1, 2000000000) = 100;
  options->Add<IntOption>(kTaskWorkersPerSearchWorkerId, -1, 128) = DEFAULT_TASK_WORKERS; // Use defined default
  options->Add<IntOption>(kMinimumWorkSizeForProcessingId, 2, 100000) = 20;
  options->Add<IntOption>(kMinimumWorkSizeForPickingId, 1, 100000) = 1;
  options->Add<IntOption>(kMinimumRemainingWorkSizeForPickingId, 0, 100000) = 20;
  options->Add<IntOption>(kMinimumWorkPerTaskForProcessingId, 1, 100000) = 8;
  options->Add<IntOption>(kIdlingMinimumWorkId, 0, 10000) = 0;
  options->Add<IntOption>(kThreadIdlingThresholdId, 0, 128) = 1;
  options->Add<StringOption>(kUCIOpponentId);
  options->Add<FloatOption>(kUCIRatingAdvId, -10000.0f, 10000.0f) = 0.0f;
  options->Add<BoolOption>(kSearchSpinBackoffId) = false;
  // --- Add options for any other features assumed from Ergodice if needed ---
  // options->Add<BoolOption>(kUseVarianceScalingId) = false;

  // --- Root Beam Search ADDED ---
  options->Add<IntOption>(kRootBeamMinWidthId, 0, 500) = 0; // Dynamic width disabled by default
  options->Add<IntOption>(kRootBeamMaxWidthId, 0, 500) = 0; // Beam disabled by default
  options->Add<IntOption>(kRootBeamUpdateThresholdId, 0, 1000000) = 100;
  options->Add<FloatOption>(kRootBeamUpdateIntervalFactorId, 1.0f, 10.0f) = 1.0f; // Default 1.0 (fixed interval)
  // --- END Root Beam Search ADDED ---

  // Hide options if desired
  options->HideOption(kNoiseEpsilonId);
  options->HideOption(kNoiseAlphaId);
  options->HideOption(kLogLiveStatsId);
  options->HideOption(kDisplayCacheUsageId);
  options->HideOption(kRootHasOwnCpuctParamsId);
  options->HideOption(kCpuctAtRootId);
  options->HideOption(kCpuctBaseAtRootId);
  options->HideOption(kCpuctFactorAtRootId);
  options->HideOption(kFpuStrategyAtRootId);
  options->HideOption(kFpuValueAtRootId);
  options->HideOption(kTemperatureId);
  options->HideOption(kTempDecayMovesId);
  options->HideOption(kTempDecayDelayMovesId);
  options->HideOption(kTemperatureCutoffMoveId);
  options->HideOption(kTemperatureEndgameId);
  options->HideOption(kTemperatureWinpctCutoffId);
  options->HideOption(kTemperatureVisitOffsetId);
  options->HideOption(kContemptMaxValueId);
  options->HideOption(kWDLContemptAttenuationId);
  options->HideOption(kWDLMaxSId);
  options->HideOption(kWDLDrawRateTargetId);
  options->HideOption(kWDLBookExitBiasId);
  // --- Hide Beam options if desired ---
  // options->HideOption(kRootBeamMinWidthId);
  // options->HideOption(kRootBeamMaxWidthId);
  // options->HideOption(kRootBeamUpdateThresholdId);
  // options->HideOption(kRootBeamUpdateIntervalFactorId);
}

SearchParams::SearchParams(const OptionsDict& options)
    : options_(options),
      kCpuct(options.Get<float>(kCpuctId)),
      kCpuctAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctAtRootId
                                                      : kCpuctId)),
      kCpuctExponent(options.Get<float>(kCpuctExponentId)),
      kCpuctExponentAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctExponentAtRootId
                                                      : kCpuctExponentId)),
      kCpuctBase(options.Get<float>(kCpuctBaseId)),
      kCpuctBaseAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctBaseAtRootId
                                                      : kCpuctBaseId)),
      kCpuctFactor(options.Get<float>(kCpuctFactorId)),
      kCpuctFactorAtRoot(options.Get<float>(
          options.Get<bool>(kRootHasOwnCpuctParamsId) ? kCpuctFactorAtRootId
                                                      : kCpuctFactorId)),
      kTwoFoldDraws(options.Get<bool>(kTwoFoldDrawsId)),
      kNoiseEpsilon(options.Get<float>(kNoiseEpsilonId)),
      kNoiseAlpha(options.Get<float>(kNoiseAlphaId)),
      kFpuAbsolute(options.Get<std::string>(kFpuStrategyId) == "absolute"),
      kFpuValue(options.Get<float>(kFpuValueId)),
      kFpuAbsoluteAtRoot(
          (options.Get<std::string>(kFpuStrategyAtRootId) == "same" &&
           kFpuAbsolute) ||
          options.Get<std::string>(kFpuStrategyAtRootId) == "absolute"),
      kFpuValueAtRoot(options.Get<std::string>(kFpuStrategyAtRootId) == "same"
                          ? kFpuValue
                          : options.Get<float>(kFpuValueAtRootId)),
      kCacheHistoryLength(options.Get<int>(kCacheHistoryLengthId)),
      kPolicySoftmaxTemp(
          options.Get<float>(SharedBackendParams::kPolicySoftmaxTemp)),
      kMaxCollisionEvents(options.Get<int>(kMaxCollisionEventsId)),
      kMaxCollisionVisits(options.Get<int>(kMaxCollisionVisitsId)),
      kOutOfOrderEval(options.Get<bool>(kOutOfOrderEvalId)),
      kStickyEndgames(options.Get<bool>(kStickyEndgamesId)),
      kSyzygyFastPlay(options.Get<bool>(kSyzygyFastPlayId)),
      kHistoryFill(EncodeHistoryFill(
          options.Get<std::string>(SharedBackendParams::kHistoryFill))),
      kMiniBatchSize(options.Get<int>(kMiniBatchSizeId)),
      kMovesLeftMaxEffect(options.Get<float>(kMovesLeftMaxEffectId)),
      kMovesLeftThreshold(options.Get<float>(kMovesLeftThresholdId)),
      kMovesLeftSlope(options.Get<float>(kMovesLeftSlopeId)),
      kMovesLeftConstantFactor(options.Get<float>(kMovesLeftConstantFactorId)),
      kMovesLeftScaledFactor(options.Get<float>(kMovesLeftScaledFactorId)),
      kMovesLeftQuadraticFactor(
          options.Get<float>(kMovesLeftQuadraticFactorId)),
      kDisplayCacheUsage(options.Get<bool>(kDisplayCacheUsageId)),
      kMaxConcurrentSearchers(options.Get<int>(kMaxConcurrentSearchersId)),
      kDrawScore(options.Get<float>(kDrawScoreId)),
      kContempt(GetContempt(options.Get<std::string>(kUCIOpponentId),
                            options.Get<std::string>(kContemptId),
                            options.Get<float>(kUCIRatingAdvId))),
      kWDLRescaleParams(
          options.Get<float>(kWDLCalibrationEloId) == 0
              ? AccurateWDLRescaleParams(
                    kContempt, options.Get<float>(kWDLDrawRateTargetId),
                    options.Get<float>(kWDLDrawRateReferenceId),
                    options.Get<float>(kWDLBookExitBiasId),
                    options.Get<float>(kContemptMaxValueId),
                    options.Get<float>(kWDLContemptAttenuationId))
              : SimplifiedWDLRescaleParams(
                    kContempt, options.Get<float>(kWDLDrawRateReferenceId),
                    options.Get<float>(kWDLCalibrationEloId),
                    options.Get<float>(kContemptMaxValueId),
                    options.Get<float>(kWDLContemptAttenuationId))),
      kWDLMaxS(options.Get<float>(kWDLMaxSId)),
      kWDLEvalObjectivity(options.Get<float>(kWDLEvalObjectivityId)),
      kMaxOutOfOrderEvalsFactor(options.Get<float>(kMaxOutOfOrderEvalsFactorId)),
      // kMaxOutOfOrderEvals initialized in constructor body
      kNpsLimit(options.Get<float>(kNpsLimitId)),
      kSolidTreeThreshold(options.Get<int>(kSolidTreeThresholdId)),
      kTaskWorkersPerSearchWorker(
          options.Get<int>(kTaskWorkersPerSearchWorkerId)),
      kMinimumWorkSizeForProcessing(
          options.Get<int>(kMinimumWorkSizeForProcessingId)),
      kMinimumWorkSizeForPicking(
          options.Get<int>(kMinimumWorkSizeForPickingId)),
      kMinimumRemainingWorkSizeForPicking(
          options.Get<int>(kMinimumRemainingWorkSizeForPickingId)),
      kMinimumWorkPerTaskForProcessing(
          options.Get<int>(kMinimumWorkPerTaskForProcessingId)),
      kIdlingMinimumWork(options.Get<int>(kIdlingMinimumWorkId)),
      kThreadIdlingThreshold(options.Get<int>(kThreadIdlingThresholdId)),
      kMaxCollisionVisitsScalingStart(
          options.Get<int>(kMaxCollisionVisitsScalingStartId)),
      kMaxCollisionVisitsScalingEnd(
          options.Get<int>(kMaxCollisionVisitsScalingEndId)),
      kMaxCollisionVisitsScalingPower(
          options.Get<float>(kMaxCollisionVisitsScalingPowerId)),
      kSearchSpinBackoff(options_.Get<bool>(kSearchSpinBackoffId)),
      // --- Add initializers for any other needed features ---
      // kUseVarianceScaling(options.Get<bool>(kUseVarianceScalingId)),

      // --- Root Beam Search ADDED ---
      kRootBeamMinWidth(options.Get<int>(kRootBeamMinWidthId)),
      kRootBeamMaxWidth(options.Get<int>(kRootBeamMaxWidthId)),
      kRootBeamUpdateThreshold(options.Get<int>(kRootBeamUpdateThresholdId)),
      kRootBeamUpdateIntervalFactor(options.Get<float>(kRootBeamUpdateIntervalFactorId))
      // --- END Root Beam Search ADDED ---
       { // Start of constructor body
           // Calculate kMaxOutOfOrderEvals here
           const int effective_batch_size = (kMiniBatchSize > 0) ? kMiniBatchSize : DEFAULT_MAX_PREFETCH; // Use a reasonable default if 0
           kMaxOutOfOrderEvals = std::max(1, static_cast<int>(kMaxOutOfOrderEvalsFactor * effective_batch_size));
       } // End of constructor body

} // namespace lczero
