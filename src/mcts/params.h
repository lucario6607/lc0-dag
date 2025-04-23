/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  ... (License header) ...
*/

#pragma once

#include "neural/encoder.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"
#include "neural/shared_params.h" // May or may not be needed

namespace lczero {

enum class ContemptMode { PLAY, WHITE, BLACK, NONE };

class SearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;

  struct WDLRescaleParams {
    WDLRescaleParams(float r, float d) { ratio = r; diff = d; }
    float ratio; float diff;
  };

  static void Populate(OptionsParser* options);

  // --- Existing Getters (ensure all needed ones are present) ---
  int GetMiniBatchSize() const { return kMiniBatchSize; }
  int GetMaxPrefetchBatch() const { return options_.Get<int>(kMaxPrefetchBatchId); }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctBase(bool at_root) const { return at_root ? kCpuctBaseAtRoot : kCpuctBase; }
  float GetCpuctExponent(bool at_root) const { return at_root ? kCpuctExponentAtRoot : kCpuctExponent; }
  float GetCpuctFactor(bool at_root) const { return at_root ? kCpuctFactorAtRoot : kCpuctFactor; }
  bool GetTwoFoldDraws() const { return kTwoFoldDraws; }
  float GetTemperature() const { return options_.Get<float>(kTemperatureId); }
  float GetTemperatureVisitOffset() const { return options_.Get<float>(kTemperatureVisitOffsetId); }
  int GetTempDecayMoves() const { return options_.Get<int>(kTempDecayMovesId); }
  int GetTempDecayDelayMoves() const { return options_.Get<int>(kTempDecayDelayMovesId); }
  int GetTemperatureCutoffMove() const { return options_.Get<int>(kTemperatureCutoffMoveId); }
  float GetTemperatureEndgame() const { return options_.Get<float>(kTemperatureEndgameId); }
  float GetTemperatureWinpctCutoff() const { return options_.Get<float>(kTemperatureWinpctCutoffId); }
  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetNoiseAlpha() const { return kNoiseAlpha; }
  bool GetVerboseStats() const { return options_.Get<bool>(kVerboseStatsId); }
  bool GetLogLiveStats() const { return options_.Get<bool>(kLogLiveStatsId); }
  bool GetFpuAbsolute(bool at_root) const { return at_root ? kFpuAbsoluteAtRoot : kFpuAbsolute; }
  float GetFpuValue(bool at_root) const { return at_root ? kFpuValueAtRoot : kFpuValue; }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicySoftmaxTemp() const { return kPolicySoftmaxTemp; }
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisits() const { return kMaxCollisionVisits; }
  bool GetOutOfOrderEval() const { return kOutOfOrderEval; }
  bool GetStickyEndgames() const { return kStickyEndgames; }
  bool GetSyzygyFastPlay() const { return kSyzygyFastPlay; }
  int GetMultiPv() const { return options_.Get<int>(kMultiPvId); }
  bool GetPerPvCounters() const { return options_.Get<bool>(kPerPvCountersId); }
  std::string GetScoreType() const { return options_.Get<std::string>(kScoreTypeId); }
  FillEmptyHistory GetHistoryFill() const { return kHistoryFill; }
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftSlope() const { return kMovesLeftSlope; }
  float GetMovesLeftConstantFactor() const { return kMovesLeftConstantFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const { return kMovesLeftQuadraticFactor; }
  bool GetDisplayCacheUsage() const { return kDisplayCacheUsage; }
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetDrawScore() const { return kDrawScore; }
  ContemptMode GetContemptMode() const; // Definition likely in .cc
  float GetWDLRescaleRatio() const { return kWDLRescaleParams.ratio; }
  float GetWDLRescaleDiff() const { return kWDLRescaleParams.diff; }
  float GetWDLMaxS() const { return kWDLMaxS; }
  float GetWDLEvalObjectivity() const { return kWDLEvalObjectivity; }
  float GetMaxOutOfOrderEvalsFactor() const { return kMaxOutOfOrderEvalsFactor; }
  uint32_t GetMaxOutOfOrderEvals() const { return kMaxOutOfOrderEvals; }
  float GetNpsLimit() const { return kNpsLimit; }
  int GetSolidTreeThreshold() const { return kSolidTreeThreshold; }
  int GetTaskWorkersPerSearchWorker() const { return kTaskWorkersPerSearchWorker; }
  int GetMinimumWorkSizeForProcessing() const { return kMinimumWorkSizeForProcessing; }
  int GetMinimumWorkSizeForPicking() const { return kMinimumWorkSizeForPicking; }
  int GetMinimumRemainingWorkSizeForPicking() const { return kMinimumRemainingWorkSizeForPicking; }
  int GetMinimumWorkPerTaskForProcessing() const { return kMinimumWorkPerTaskForProcessing; }
  int GetIdlingMinimumWork() const { return kIdlingMinimumWork; }
  int GetThreadIdlingThreshold() const { return kThreadIdlingThreshold; }
  int GetMaxCollisionVisitsScalingStart() const { return kMaxCollisionVisitsScalingStart; }
  int GetMaxCollisionVisitsScalingEnd() const { return kMaxCollisionVisitsScalingEnd; }
  float GetMaxCollisionVisitsScalingPower() const { return kMaxCollisionVisitsScalingPower; }
  bool GetSearchSpinBackoff() const { return kSearchSpinBackoff; }
  // --- Add Getters for any other features assumed from Ergodice if needed ---
  // e.g., bool GetUseVarianceScaling() const { return kUseVarianceScaling; }


  // --- Root Beam Search Getters ---
  int GetRootBeamMinWidth() const { return kRootBeamMinWidth; }
  int GetRootBeamMaxWidth() const { return kRootBeamMaxWidth; }
  int GetRootBeamUpdateThreshold() const { return kRootBeamUpdateThreshold; }
  float GetRootBeamUpdateIntervalFactor() const { return kRootBeamUpdateIntervalFactor; }
  // --- END Root Beam Search Getters ---

  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kMaxPrefetchBatchId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctExponentId;
  static const OptionId kCpuctExponentAtRootId;
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctBaseAtRootId;
  static const OptionId kCpuctFactorId;
  static const OptionId kCpuctFactorAtRootId;
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kTwoFoldDrawsId;
  static const OptionId kTemperatureId;
  static const OptionId kTempDecayMovesId;
  static const OptionId kTempDecayDelayMovesId;
  static const OptionId kTemperatureCutoffMoveId;
  static const OptionId kTemperatureEndgameId;
  static const OptionId kTemperatureWinpctCutoffId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kNoiseEpsilonId;
  static const OptionId kNoiseAlphaId;
  static const OptionId kVerboseStatsId;
  static const OptionId kLogLiveStatsId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuValueId;
  static const OptionId kFpuStrategyAtRootId;
  static const OptionId kFpuValueAtRootId;
  static const OptionId kCacheHistoryLengthId;
  // static const OptionId kPolicySoftmaxTempId; // Assumed shared
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kOutOfOrderEvalId;
  static const OptionId kStickyEndgamesId;
  static const OptionId kSyzygyFastPlayId;
  static const OptionId kMultiPvId;
  static const OptionId kPerPvCountersId;
  static const OptionId kScoreTypeId;
  // static const OptionId kHistoryFillId; // Assumed shared
  static const OptionId kMovesLeftMaxEffectId;
  static const OptionId kMovesLeftThresholdId;
  static const OptionId kMovesLeftConstantFactorId;
  static const OptionId kMovesLeftScaledFactorId;
  static const OptionId kMovesLeftQuadraticFactorId;
  static const OptionId kMovesLeftSlopeId;
  static const OptionId kDisplayCacheUsageId;
  static const OptionId kMaxConcurrentSearchersId;
  static const OptionId kDrawScoreId;
  static const OptionId kContemptModeId;
  static const OptionId kContemptId;
  static const OptionId kContemptMaxValueId;
  static const OptionId kWDLCalibrationEloId;
  static const OptionId kWDLContemptAttenuationId;
  static const OptionId kWDLMaxSId;
  static const OptionId kWDLEvalObjectivityId;
  static const OptionId kWDLDrawRateTargetId;
  static const OptionId kWDLDrawRateReferenceId;
  static const OptionId kWDLBookExitBiasId;
  static const OptionId kMaxOutOfOrderEvalsFactorId;
  static const OptionId kNpsLimitId;
  static const OptionId kSolidTreeThresholdId;
  static const OptionId kTaskWorkersPerSearchWorkerId;
  static const OptionId kMinimumWorkSizeForProcessingId;
  static const OptionId kMinimumWorkSizeForPickingId;
  static const OptionId kMinimumRemainingWorkSizeForPickingId;
  static const OptionId kMinimumWorkPerTaskForProcessingId;
  static const OptionId kIdlingMinimumWorkId;
  static const OptionId kThreadIdlingThresholdId;
  static const OptionId kMaxCollisionVisitsScalingStartId;
  static const OptionId kMaxCollisionVisitsScalingEndId;
  static const OptionId kMaxCollisionVisitsScalingPowerId;
  static const OptionId kUCIOpponentId;
  static const OptionId kUCIRatingAdvId;
  static const OptionId kSearchSpinBackoffId;
  // --- Add OptionIds for any other features assumed from Ergodice if needed ---
  // static const OptionId kUseVarianceScalingId;


  // --- Root Beam Search ADDED ---
  static const OptionId kRootBeamMinWidthId;
  static const OptionId kRootBeamMaxWidthId;
  static const OptionId kRootBeamUpdateThresholdId;
  static const OptionId kRootBeamUpdateIntervalFactorId;
  // --- END Root Beam Search ADDED ---

 private:
  const OptionsDict& options_;
  // Cached parameter values.
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctExponent;
  const float kCpuctExponentAtRoot;
  const float kCpuctBase;
  const float kCpuctBaseAtRoot;
  const float kCpuctFactor;
  const float kCpuctFactorAtRoot;
  const bool kTwoFoldDraws;
  const float kNoiseEpsilon;
  const float kNoiseAlpha;
  const bool kFpuAbsolute;
  const float kFpuValue;
  const bool kFpuAbsoluteAtRoot;
  const float kFpuValueAtRoot;
  const int kCacheHistoryLength;
  const float kPolicySoftmaxTemp;
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kOutOfOrderEval;
  const bool kStickyEndgames;
  const bool kSyzygyFastPlay;
  const FillEmptyHistory kHistoryFill;
  const int kMiniBatchSize;
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftSlope;
  const float kMovesLeftConstantFactor;
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const bool kDisplayCacheUsage;
  const int kMaxConcurrentSearchers;
  const float kDrawScore;
  const float kContempt;
  const WDLRescaleParams kWDLRescaleParams;
  const float kWDLMaxS;
  const float kWDLEvalObjectivity;
  const float kMaxOutOfOrderEvalsFactor;
  uint32_t kMaxOutOfOrderEvals; // Made non-const
  const float kNpsLimit;
  const int kSolidTreeThreshold;
  const int kTaskWorkersPerSearchWorker;
  const int kMinimumWorkSizeForProcessing;
  const int kMinimumWorkSizeForPicking;
  const int kMinimumRemainingWorkSizeForPicking;
  const int kMinimumWorkPerTaskForProcessing;
  const int kIdlingMinimumWork;
  const int kThreadIdlingThreshold;
  const int kMaxCollisionVisitsScalingStart;
  const int kMaxCollisionVisitsScalingEnd;
  const float kMaxCollisionVisitsScalingPower;
  const bool kSearchSpinBackoff;
  // --- Add const members for any other features assumed from Ergodice if needed ---
  // const bool kUseVarianceScaling;

  // --- Root Beam Search ADDED ---
  const int kRootBeamMinWidth;
  const int kRootBeamMaxWidth;
  const int kRootBeamUpdateThreshold;
  const float kRootBeamUpdateIntervalFactor;
  // --- END Root Beam Search ADDED ---

};

} // No classic namespace
}  // namespace lczero
