/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors
  (License header remains the same)
*/

#include "mcts/search.h" // Use correct path

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>        // Needed for sqrt, pow, log
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>      // For std::iota if used elsewhere
#include <sstream>
#include <thread>
#include <vector>

#include "mcts/node.h" // Use correct path
#include "mcts/params.h" // Use correct path
#include "mcts/stoppers/timemgr.h" // Use correct path
#include "neural/encoder.h"
#include "neural/network.h"
#include "neural/network_computation.h"
#include "syzygy/syzygy.h"
#include "utils/fastmath.h"
#include "utils/random.h"
#include "utils/spinhelper.h"

namespace lczero { // No classic namespace

namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;

MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          false, &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}

class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  // Constructor now takes mlh_enabled flag based on network capabilities
  MEvaluator(const SearchParams& params, const Node* parent = nullptr, bool mlh_enabled = true)
      : enabled_{mlh_enabled && params.GetMovesLeftMaxEffect() > 0.0f}, // Check param
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  // Calculates the utility for favoring shorter wins and longer losses.
  float GetMUtility(Node* child, float q) const {
    if (!enabled_ || !parent_within_threshold_ || !child) return 0.0f; // Add null check for child
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      q = std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
    }
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetMUtility(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    if (child.GetN() == 0) return GetDefaultMUtility();
    return GetMUtility(child.node(), q);
  }

  // The M utility to use for unvisited nodes.
  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
    return std::abs(parent->GetQ(0.0f)) > q_threshold;
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

// --- Moved Helper functions outside anonymous namespace ---
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  // Avoid division by zero or sqrt of zero if GetVisitedPolicy is 0
  const float visited_pol_sqrt = std::sqrt(std::max(0.0f, node->GetVisitedPolicy()));
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) - value * visited_pol_sqrt;
}

// Faster version for if visited_policy is readily available already.
inline float GetFpu(const SearchParams& params, Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) - value * std::sqrt(std::max(0.0f, visited_pol));
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  // Use CpuctExponent here
  // Ensure CpuctExponent getter exists in params.h
  return (init + (k ? k * FastLog((N + base) / base) : 0.0f)) *
         std::pow(std::max((float)N, 1e-5f), params.GetCpuctExponent(is_root_node));
}

// Helper function to calculate PUCT score
inline float CalculatePuctScore(const SearchParams& params, Node* parent, const EdgeAndNode& child_edge,
                                float fpu, float draw_score, const MEvaluator& m_evaluator, float u_coeff_numerator) {
    const float Q = child_edge.GetQ(fpu, draw_score);
    // Use GetU which implicitly uses N_child in denominator
    const float U = child_edge.GetU(u_coeff_numerator);
    const float M = m_evaluator.GetMUtility(child_edge, Q);
    return Q + U + M;
}
// --- End Moved Helper functions ---

}  // namespace

Search::Search(const NodeTree& tree, Network* network,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               bool ponder, const OptionsDict& options,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite && !ponder),
      stopper_(std::move(stopper)),
      root_node_(tree.GetCurrentHead()),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      network_(network),
      network_capabilities_(network->GetCapabilities()),
      params_(options),
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_->GetN()),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {
  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  contempt_mode_ = params_.GetContemptMode();
  // Make sure the contempt mode is never "play" beyond this point.
  if (contempt_mode_ == ContemptMode::PLAY) {
    if (infinite) {
      contempt_mode_ = ContemptMode::NONE;
      if (params_.GetWDLRescaleDiff() != 0.0f) {
        std::vector<ThinkingInfo> info(1);
        info.back().comment =
            "WARNING: Contempt mode set to 'disable' as 'play' not supported "
            "for infinite search.";
        uci_responder_->OutputThinkingInfo(&info);
      }
    } else {
      contempt_mode_ = played_history_.IsBlackToMove() != ponder
                           ? ContemptMode::BLACK
                           : ContemptMode::WHITE;
    }
  }
  root_beam_active_ = false;
  last_root_beam_update_visits_ = 0;
  last_root_beam_interval_used_ = params_.GetRootBeamUpdateThreshold();
}

namespace { // Keep ApplyDirichletNoise local
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace

// --- Root Beam Search Update Function ---
void Search::UpdateRootBeam(Node* root_node) REQUIRES(nodes_mutex_) {
    const int min_k = params_.GetRootBeamMinWidth();
    const int max_k = params_.GetRootBeamMaxWidth();
    // If min width >= max width, dynamic width is disabled.
    bool use_dynamic_k = (min_k > 0 && min_k < max_k);
    int current_k = max_k; // Default to max width, may be adjusted by dynamic logic

    if (!root_node->HasChildren() || max_k <= 0) { // Use MaxWidth to enable/disable
        root_beam_indices_.clear();
        root_beam_active_ = false; // Ensure beam is inactive if conditions not met
        return;
    }

    const int num_children = root_node->GetNumEdges(); // Use GetNumEdges as it reflects potential children


    if (num_children <= (use_dynamic_k ? min_k : max_k)) { // Use min_k if dynamic, else max_k
        // Beam is wider than or equal to number of children, no restriction needed
        root_beam_indices_.clear(); // Signal no restriction
        root_beam_active_ = false; // Keep inactive if no restriction needed
        return;
    }

    // *** CHANGE: Use PUCT score for ranking ***
    std::vector<std::pair<float, int>> scored_children; // {PUCT_score, index}
    scored_children.reserve(num_children);

    // Calculate PUCT scores for ranking
    const float draw_score = GetDrawScore(/* is_odd_depth= */ false); // Root is even depth
    const float fpu = GetFpu(params_, root_node, /* is_root= */ true, draw_score);
    const MEvaluator m_evaluator = network_capabilities_.has_mlh ? MEvaluator(params_, root_node, network_capabilities_.has_mlh) : MEvaluator(params_, root_node, false);
    const float cpuct_val = ComputeCpuct(params_, root_node->GetN(), true);
    // Calculate U coefficient numerator (sqrt(N_parent))
    // Note: GetChildrenVisits might be slightly inaccurate if some visits ended without children, GetN() is safer
    const float u_coeff_numerator = cpuct_val * std::sqrt(std::max((float)root_node->GetN(), 1.0f));

    int idx = 0;
    for (const auto& edge : root_node->Edges()) {
        // Skip moves filtered by UCI searchmoves or TB
        if (!root_move_filter_.empty() &&
            std::find(root_move_filter_.begin(), root_move_filter_.end(),
                      edge.GetMove()) == root_move_filter_.end()) {
             idx++;
             continue;
        }
        // Skip nodes with virtual loss - PUCT score calculation implicitly handles NInFlight via GetNStarted in GetU

        // *** CHANGE: Calculate and store PUCT score instead of visits ***
        float puct_score = CalculatePuctScore(params_, root_node, edge, fpu, draw_score, m_evaluator, u_coeff_numerator);
        scored_children.push_back({puct_score, idx++});
    }

    if (scored_children.empty()) { // Handle case where all moves were filtered
         root_beam_indices_.clear();
         root_beam_active_ = false;
         return;
    }

    // *** CHANGE: Sort descending by PUCT score ***
    std::sort(scored_children.rbegin(), scored_children.rend());

    // --- Interpolated Dynamic Width Heuristic (using PUCT scores) ---
    // (This section remains the same as previous version, but now uses sorted scores)
    if (use_dynamic_k && scored_children.size() >= 2) {
        const float score1 = scored_children[0].first;
        const float score2 = scored_children[1].first;
        // Use effective max_k based on available moves if fewer than max_k
        const int effective_max_k_idx = std::min((int)scored_children.size() - 1, max_k - 1);
        const float score_k_max = scored_children[effective_max_k_idx].first;

        const float gap1 = score1 - score2;
        // Ensure gap_k is positive and non-zero for division
        const float gap_k = std::max(1e-6f, score1 - score_k_max);
        // Calculate relative gap and clamp between 0 and 1
        const float clamped_relative_gap = std::clamp(gap1 / gap_k, 0.0f, 1.0f);

        // Interpolate k based on the relative gap
        current_k = static_cast<int>(std::round(min_k + (max_k - min_k) * (1.0f - clamped_relative_gap)));

        LOGFILE << "Dynamic beam: Gap1=" << gap1 << ", GapK=" << gap_k << ", RelGap=" << clamped_relative_gap << ", InterpolatedK=" << current_k;

        // Clamp k to the number of available ranked children and within Min/Max bounds
        current_k = std::min(current_k, (int)scored_children.size());
        current_k = std::max(current_k, min_k);
        current_k = std::min(current_k, max_k);

    } else {
        // Use fixed MaxWidth if dynamic k is disabled or not enough moves
        current_k = std::min(max_k, (int)scored_children.size());
    }
    // --- End Dynamic Width Heuristic ---


    // Store the indices of the top k children (based on sorted scores)
    root_beam_indices_.clear();
    root_beam_indices_.reserve(current_k);
    for (int i = 0; i < current_k; ++i) {
        root_beam_indices_.push_back(scored_children[i].second);
    }
    root_beam_active_ = true; // Activate the beam
    LOGFILE << "Root beam updated. Width=" << current_k << ". Top " << root_beam_indices_.size() << " indices selected.";
}
// --- End Root Beam Search Implementation ---

namespace { // Keep WDLRescale local
// WDL conversion formula based on random walk model.
inline double WDLRescale(float& v, float& d, float wdl_rescale_ratio,
                         float wdl_rescale_diff, float sign, bool invert,
                         float max_reasonable_s) {
  if (invert) {
    wdl_rescale_diff = -wdl_rescale_diff;
    wdl_rescale_ratio = 1.0f / wdl_rescale_ratio;
  }
  auto w = (1 + v - d) / 2;
  auto l = (1 - v - d) / 2;
  // Safeguard against numerical issues; skip WDL transformation if WDL is too
  // extreme.
  const float eps = 0.0001f;
  if (w > eps && d > eps && l > eps && w < (1.0f - eps) && d < (1.0f - eps) &&
      l < (1.0f - eps)) {
    auto a = FastLog(1 / l - 1);
    auto b = FastLog(1 / w - 1);
    auto s = 2 / (a + b);
    // Safeguard against unrealistically broad WDL distributions coming from
    // the NN. Originally hardcoded, made into a parameter for piece odds.
    if (!invert) s = std::min(max_reasonable_s, s);
    auto mu = (a - b) / (a + b);
    auto s_new = s * wdl_rescale_ratio;
    if (invert) {
      std::swap(s, s_new);
      s = std::min(max_reasonable_s, s);
    }
    auto mu_new = mu + sign * s * s * wdl_rescale_diff;
    auto w_new = FastLogistic((-1.0f + mu_new) / s_new);
    auto l_new = FastLogistic((-1.0f - mu_new) / s_new);
    v = w_new - l_new;
    d = std::max(0.0f, 1.0f - w_new - l_new);
    return mu_new;
  }
  return 0;
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  const auto edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const bool display_cache_usage = params_.GetDisplayCacheUsage();
  const auto draw_score = GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();

  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
       common_info.nps = total_playouts_ * 1000 / time_since_first_batch_ms;
    }
  }
  if (display_cache_usage) {
     // Requires access to the cache, assuming search_ holds it or can get it
     // common_info.hashfull = ...; // Placeholder
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = -root_node_->GetQ(-draw_score);
  const auto default_wl = -root_node_->GetWL();
  const auto default_d = root_node_->GetD();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    auto wl = edge.GetWL(default_wl);
    auto d = edge.GetD(default_d);
    float mu_uci = 0.0f;
    if (score_type == "WDL_mu" || (params_.GetWDLRescaleDiff() != 0.0f &&
                                   contempt_mode_ != ContemptMode::NONE)) {
      auto sign = ((contempt_mode_ == ContemptMode::BLACK) ==
                   played_history_.IsBlackToMove())
                      ? 1.0f
                      : -1.0f;
      mu_uci = WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          sign, true, params_.GetWDLMaxS());
    }
    const auto q = edge.GetQ(default_q, draw_score);
    if (edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f)) / 2 + (edge.IsTbTerminal() ? 101 : 1),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    } else if (score_type == "WDL_mu") {
      const float centipawn_fallback_threshold = 0.996f;
      float centipawn_score = 45 * tan(1.56728071628 * wl);
      uci_info.score =
          network_capabilities_.has_wdl && mu_uci != 0.0f &&
                  std::abs(wl) + d < centipawn_fallback_threshold &&
                  (std::abs(mu_uci) < 1.0f ||
                   std::abs(centipawn_score) < std::abs(100 * mu_uci))
              ? 100 * mu_uci
              : centipawn_score;
    }

    auto wdl_w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - d))));
    auto wdl_l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - d))));
    auto wdl_d = 1000 - wdl_w - wdl_l;
    if (wdl_d < 0) {
      wdl_w = std::min(1000, std::max(0, wdl_w + wdl_d / 2));
      wdl_l = 1000 - wdl_w;
      wdl_d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{wdl_w, wdl_d, wdl_l};
    if (network_capabilities_.has_mlh) {
      uci_info.moves_left = static_cast<int>(
          (1.0f + edge.GetM(1.0f + root_node_->GetM())) / 2.0f);
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;
      depth += 1;
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    last_outputted_info_edge_ = current_best_edge_.edge();
  }

  uci_responder_->OutputThinkingInfo(&uci_infos);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
      uci_responder_->OutputThinkingInfo(&info);
    }
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_) {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetDrawScore()
              : -params_.GetDrawScore());
}

std::vector<std::string> Search::GetVerboseStats(Node* node) const {
  assert(node == root_node_ || node->GetParent() == root_node_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetN(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  std::vector<EdgeAndNode> edges;
  for (const auto& edge : node->Edges()) edges.push_back(edge);

  std::sort(edges.begin(), edges.end(),
            [&fpu, &U_coeff, &draw_score](EdgeAndNode a, EdgeAndNode b) {
              return std::forward_as_tuple(
                         a.GetN(), a.GetQ(fpu, draw_score) + a.GetU(U_coeff)) <
                     std::forward_as_tuple(
                         b.GetN(), b.GetQ(fpu, draw_score) + b.GetU(U_coeff));
            });

  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, p >= 0.99995f ? 1 : 2);
  };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      auto wl = sign * n->GetWL();
      auto d = n->GetD();
      auto is_perspective = ((contempt_mode_ == ContemptMode::BLACK) ==
                             played_history_.IsBlackToMove())
                                ? 1.0f
                                : -1.0f;
      WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          is_perspective, true, params_.GetWDLMaxS());
      print(oss, "(WL: ", wl, ") ", 8, 5);
      print(oss, "(D: ", d, ") ", 5, 3);
      print(oss, "(M: ", n->GetM(), ") ", 4, 1);
      print(oss, "(Q: ", wl + draw_score * d, ") ", 8, 5);
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
      print(oss, "(Q: ", fpu, ") ", 8, 5);
    }
  };
  auto print_tail = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    std::optional<float> v;
    // --- Use GetCachedNNEval for V ---
    std::optional<Network::Output> nn_eval_opt; // Use Network::Output
    if (n && !n->IsTerminal()) {
         nn_eval_opt = GetCachedNNEval(n); // Use the class method
    }
    if (n && n->IsTerminal()) {
      v = n->GetQ(sign * draw_score);
    } else if (nn_eval_opt) {
        v = -nn_eval_opt->q; // Use the value from the cached output
    }
    // --- End Use GetCachedNNEval ---
    if (v) {
      print(oss, "(V: ", sign * *v, ") ", 7, 4);
    } else {
      *oss << "(V:  -.----) ";
    }

    if (n) {
      auto [lo, up] = n->GetBounds();
      if (sign == -1) {
        lo = -lo;
        up = -up;
        std::swap(lo, up);
      }
      *oss << (lo == up                                                ? "(T) "
               : lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) "
               : lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) "
                                                                       : "");
    }
  };

  std::vector<std::string> infos;
  const auto m_evaluator =
      network_capabilities_.has_mlh ? MEvaluator(params_, node, network_capabilities_.has_mlh) : MEvaluator(params_, node, false); // Use network_capabilities_
  for (const auto& edge : edges) {
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetMUtility(edge, Q);
    std::ostringstream oss;
    oss << std::left;
    // TODO: should this be displaying transformed index?
    print_head(&oss, edge.GetMove(is_black_to_move).ToString(true),
               MoveToNNIndex(edge.GetMove(), 0), edge.GetN(),
               edge.GetNInFlight(), edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", edge.GetU(U_coeff), ") ", 6, 5);
    print(&oss, "(S: ", Q + edge.GetU(U_coeff) + M, ") ", 8, 5);
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }

  // Include stats about the node in similar format to its children above.
  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node);
  infos.emplace_back(oss.str());
  return infos;
}

void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  for (auto& edge : root_node_->Edges()) {
    if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) {
      continue;
    }
    if (edge.HasNode()) {
      LOGFILE << "--- Opponent moves after: " << final_bestmove_.ToString(true);
      for (const auto& line : GetVerboseStats(edge.node())) {
        LOGFILE << line;
      }
    }
  }
}

PositionHistory Search::GetPositionHistoryAtNode(const Node* node) const {
  PositionHistory history(played_history_);
  std::vector<Move> rmoves;
  for (const Node* n = node; n != root_node_; n = n->GetParent()) {
      if (!n->GetParent()) break; // Stop if parent is null (should only happen for root)
      Edge* own_edge = n->GetOwnEdge();
      if (!own_edge) break; // Should not happen for non-root nodes
      rmoves.push_back(own_edge->GetMove());
  }
  for (auto it = rmoves.rbegin(); it != rmoves.rend(); it++) {
    history.Append(*it);
  }
  return history;
}

namespace {
std::vector<Move> GetNodeLegalMoves(const Node* node, const ChessBoard& board) {
  if (!node) return {};
  std::vector<Move> moves;
  if (node && node->HasChildren()) {
    moves.reserve(node->GetNumEdges());
    std::transform(node->Edges().begin(), node->Edges().end(),
                   std::back_inserter(moves),
                   [](const auto& edge) { return edge.GetMove(); });
    return moves;
  }
  return board.GenerateLegalMoves();
}
}  // namespace

// Adjusted return type to match Network::Output
std::optional<Network::Output> Search::GetCachedNNEval(const Node* node) const {
  if (!node) return {};
  PositionHistory history = GetPositionHistoryAtNode(node);
  std::vector<Move> legal_moves =
      GetNodeLegalMoves(node, history.Last().GetBoard());
  // Use network_->GetCachedOutput instead of backend_->GetCachedEvaluation
  return network_->GetCachedOutput(
      EvalPosition{history.GetPositions(), legal_moves});
}

void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  if (params_.GetNpsLimit() > 0) {
    hints->UpdateEstimatedNps(params_.GetNpsLimit());
  }
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (total_playouts_ + initial_visits_ == 0) return;

  if (!stop_.load(std::memory_order_acquire)) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }

  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  }
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();
  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_, 0);
  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());
  if (is_terminal) *is_terminal = best_edge.IsTerminal();
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.GetM(parent_m - 1) + 1};
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}

// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (root_node_->GetN() == 0) return;
  if (!root_node_->HasChildren()) return;

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;

  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature && decay_moves) {
    if (moves >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves >= decay_delay_moves) {
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  auto bestmove_edge = temperature
                           ? GetBestRootChildWithTemperature(temperature)
                           : GetBestChildNoTemperature(root_node_, 0);
  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());

  if (bestmove_edge.GetN() > 0 && bestmove_edge.node() && bestmove_edge.node()->HasChildren()) { // Added null check for node()
    final_pondermove_ = GetBestChildNoTemperature(bestmove_edge.node(), 1)
                            .GetMove(!played_history_.IsBlackToMove());
  } else {
    final_pondermove_ = Move::kNullMove; // Set ponder to null if no valid child
  }
}

// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  // Even if Edges is populated at this point, its a race condition to access
  // the node, so exit quickly.
  if (parent->GetN() == 0) return {};
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);
  // Best child is selected using the following criteria:
  // * Prefer shorter terminal wins / avoid shorter terminal losses.
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::vector<EdgeAndNode> edges;
  for (auto& edge : parent->Edges()) {
    if (parent == root_node_ && !root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    edges.push_back(edge);
  }
  const auto middle = (static_cast<int>(edges.size()) > count)
                          ? edges.begin() + count
                          : edges.end();
  std::partial_sort(
      edges.begin(), middle, edges.end(),
      [draw_score](const auto& a, const auto& b) {
        // The function returns "true" when a is preferred to b.

        // Lists edge types from less desirable to more desirable.
        enum EdgeRank {
          kTerminalLoss,
          kTablebaseLoss,
          kNonTerminal,  // Non terminal or terminal draw.
          kTablebaseWin,
          kTerminalWin,
        };

        auto GetEdgeRank = [](const EdgeAndNode& edge) {
          // This default isn't used as wl only checked for case edge is
          // terminal.
          const auto wl = edge.GetWL(0.0f);
          // Not safe to access IsTerminal if GetN is 0.
          if (edge.GetN() == 0 || !edge.IsTerminal() || !wl) {
            return kNonTerminal;
          }
          if (edge.IsTbTerminal()) {
            return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
          }
          return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };

        // If moves have different outcomes, prefer better outcome.
        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;

        // If both are terminal draws, try to make it shorter.
        // Not safe to access IsTerminal if GetN is 0.
        if (a_rank == kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) {
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            // Prefer non-tablebase draws.
            return a.IsTbTerminal() < b.IsTbTerminal();
          }
          // Prefer shorter draws.
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Neither is terminal, use standard rule.
        if (a_rank == kNonTerminal) {
          // Prefer largest playouts then eval then prior.
          if (a.GetN() != b.GetN()) return a.GetN() > b.GetN(); // Use GetN for sorting
          // Default doesn't matter here so long as they are the same as either
          // both are N==0 (thus we're comparing equal defaults) or N!=0 and
          // default isn't used.
          if (a.GetQ(0.0f, draw_score) != b.GetQ(0.0f, draw_score)) {
            return a.GetQ(0.0f, draw_score) > b.GetQ(0.0f, draw_score);
          }
          return a.GetP() > b.GetP();
        }

        // Both variants are winning, prefer shortest win.
        if (a_rank > kNonTerminal) {
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Both variants are losing, prefer longest losses.
        return a.GetM(0.0f) > b.GetM(0.0f);
      });

  if (count < static_cast<int>(edges.size())) {
    edges.resize(count);
  }
  return edges;
}

// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}

// Returns a child of a root chosen according to weighted-by-temperature visit
// count.
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const {
  // Root is at even depth.
  const float draw_score = GetDrawScore(/* is_odd_depth= */ false);

  std::vector<float> cumulative_sums;
  float sum = 0.0;
  float max_n = 0.0; // Changed from max_weight to max_n
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -1.0f;
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);

  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetN() + offset > max_n) { // Changed from GetWeight to GetN
      max_n = edge.GetN() + offset;
      max_eval = edge.GetQ(fpu, draw_score);
    }
  }

  // TODO(crem) Simplify this code when samplers.h is merged.
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    sum += std::pow(
        std::max(0.0f,
                 (max_n <= 0.0f
                      ? edge.GetP()
                      : ((static_cast<float>(edge.GetN()) + offset) / max_n))), // Changed GetWeight to GetN
        1 / temperature);
    cumulative_sums.push_back(sum);
  }
  // Check if sum is positive, handle case where all moves might be filtered
  if (sum <= 0.0f) {
      LOGFILE << "Warning: No valid moves found for temperature selection. Falling back.";
      return GetBestChildNoTemperature(root_node_, 0); // Fallback
  }

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int idx =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  // Need to re-filter here as cumulative_sums only contains allowed moves
  int current_filtered_idx = 0;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) continue;
    if (current_filtered_idx++ == idx) return edge;
  }

  // Should be unreachable if sum > 0
  LOGFILE << "Error: Temperature selection failed to find move.";
  // Fallback: return best child without temperature
  return GetBestChildNoTemperature(root_node_, 0);
}

void Search::StartThreads(size_t how_many) {
  Mutex::Lock lock(threads_mutex_);
  if (how_many == 0 && threads_.size() == 0) {
    how_many = network_capabilities_.suggested_num_search_threads + // Use network_capabilities_
               !network_capabilities_.runs_on_cpu; // Use network_capabilities_
  }
  thread_count_.store(how_many, std::memory_order_release);
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  for (size_t i = 0; i < how_many; i++) {
    threads_.emplace_back([this]() {
      SearchWorker worker(this, params_);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}

void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  {
    Mutex::Lock counters_lock(counters_mutex_);
    stats->time_since_first_batch = GetTimeSinceFirstBatch();
    if (!nps_start_time_ && total_playouts_ > 0) {
      nps_start_time_ = std::chrono::steady_clock::now();
    }
  }
  stats->total_nodes = total_playouts_ + initial_visits_; // Changed from total_visits
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;
  stats->mate_depth = std::numeric_limits<int>::max();

  // If root node hasn't finished first visit, none of this code is safe.
  if (root_node_->GetN() > 0) {
    const auto draw_score = GetDrawScore(true);
    const float fpu =
        GetFpu(params_, root_node_, /* is_root_node */ true, draw_score);
    float max_q_plus_m = -1000;
    uint64_t max_n = 0;
    bool max_n_has_max_q_plus_m = true;
    const auto m_evaluator = network_capabilities_.has_mlh // Use network_capabilities_
                                 ? MEvaluator(params_, root_node_, network_capabilities_.has_mlh)
                                 : MEvaluator(params_, root_node_, false);
    for (const auto& edge : root_node_->Edges()) {
      const auto n = edge.GetN();
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetMUtility(edge, q);
      const auto q_plus_m = q + m;
      stats->edge_n.push_back(n);
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) {
        stats->win_found = true;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) {
        stats->num_losing_edges += 1;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) == 1.0f &&
          !edge.IsTbTerminal()) {
        stats->mate_depth =
            std::min(stats->mate_depth,
                     static_cast<int>(std::round(edge.GetM(0.0f))) / 2 + 1);
      }

      // If game is resignable, no need for moving quicker. This allows
      // proving mate when losing anyway for better score output.
      // Hardcoded resign threshold, because there is no available parameter.
      if (n > 0 && q > -0.98f) {
        stats->may_resign = false;
      }
      if (max_n < n) {
        max_n = n;
        max_n_has_max_q_plus_m = false;
      }
      if (max_q_plus_m <= q_plus_m) {
        max_n_has_max_q_plus_m = (max_n == n);
        max_q_plus_m = q_plus_m;
      }
    }
    if (!max_n_has_max_q_plus_m) {
      stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}

void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats stats;
  while (true) {
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints);
    MaybeOutputInfo();

    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;

    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    if (remaining_time > kMaxWaitTimeMs) remaining_time = kMaxWaitTimeMs;
    if (remaining_time < kMinWaitTimeMs) remaining_time = kMinWaitTimeMs;
    // There is no real need to have max wait time, and sometimes it's fine
    // to wait without timeout at all (e.g. in `go nodes` mode), but we
    // still limit wait time for exotic cases like when pc goes to sleep
    // mode during thinking.
    // Minimum wait time is there to prevent busy wait and other threads
    // starvation.
    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    Node* node = entry.first;
    for (node = node->GetParent(); node != root_node_->GetParent();
         node = node->GetParent()) {
        if (!node) break; // Added safety check
        node->CancelScoreUpdate(entry.second);
    }
  }
  shared_collisions_.clear();
}

Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();
  }
  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

// (SearchWorker implementation - includes beam trigger in PickNodesToExtend
// and beam usage in PickNodesToExtendTask - remains the same as Answer #11,
// just without the classic:: namespace)

void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    {
      int spins = 0;
      while (true) {
        int nta = tasks_taken_.load(std::memory_order_acquire);
        int tc = task_count_.load(std::memory_order_acquire);
        if (nta < tc) {
          int val = 0;
          if (task_taking_started_.compare_exchange_weak(
                  val, 1, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
            nta = tasks_taken_.load(std::memory_order_acquire);
            tc = task_count_.load(std::memory_order_acquire);
            // We got the spin lock, double check we're still in the clear.
            if (nta < tc) {
              id = tasks_taken_.fetch_add(1, std::memory_order_acq_rel);
              task = &picking_tasks_[id];
              task_taking_started_.store(0, std::memory_order_release);
              break;
            }
            task_taking_started_.store(0, std::memory_order_release);
          }
          SpinloopPause();
          spins = 0;
          continue;
        } else if (tc != -1) {
          spins++;
          if (spins >= 512) {
            std::this_thread::yield();
            spins = 0;
          } else {
            SpinloopPause();
          }
          continue;
        }
        spins = 0;
        // Looks like sleep time.
        Mutex::Lock lock(picking_tasks_mutex_);
        // Refresh them now we have the lock.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (tc != -1) continue;
        if (nta >= tc && exiting_) return;
        task_added_.wait(lock.get_raw());
        // And refresh again now we're awake.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (nta >= tc && exiting_) return;
      }
    }
    if (task != nullptr) {
      switch (task->task_type) {
        case PickTask::kGathering: {
          PickNodesToExtendTask(task->start, task->base_depth,
                                task->collision_limit, task->moves_to_base,
                                &(task->results), &(task_workspaces_[tid]));
          break;
        }
        case PickTask::kProcessing: {
          ProcessPickedTask(task->start_idx, task->end_idx,
                            &(task_workspaces_[tid]));
          break;
        }
      }
      picking_tasks_[id].complete = true;
      completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration(search_->network_->CreateComputation());

  if (params_.GetMaxConcurrentSearchers() != 0) {
    std::unique_ptr<SpinHelper> spin_helper;
    if (params_.GetSearchSpinBackoff()) {
      spin_helper = std::make_unique<ExponentialBackoffSpinHelper>();
    } else {
      spin_helper = std::make_unique<SpinHelper>();
    }

    while (true) {
      if (search_->stop_.load(std::memory_order_acquire) &&
          search_->GetTotalPlayouts() + search_->initial_visits_ > 0) {
        return;
      }
      int available =
          search_->pending_searchers_.load(std::memory_order_acquire);
      if (available == 0) {
        spin_helper->Wait();
        continue;
      }
      if (search_->pending_searchers_.compare_exchange_weak(
              available, available - 1, std::memory_order_acq_rel)) {
        break;
      } else {
        spin_helper->Backoff();
      }
    }
  }

  // 2. Gather minibatch.
  GatherMinibatch();
  task_count_.store(-1, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);

  // 2b. Collect collisions.
  CollectCollisions();

  // 3. Prefetch into cache.
  MaybePrefetchIntoCache();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 4. Run NN computation.
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();

  // If required, waste time to limit nps.
  if (params_.GetNpsLimit() > 0) {
    while (search_->IsSearchActive()) {
      int64_t time_since_first_batch_ms = 0;
      {
        Mutex::Lock lock(search_->counters_mutex_);
        time_since_first_batch_ms = search_->GetTimeSinceFirstBatch();
      }
      if (time_since_first_batch_ms <= 0) {
        time_since_first_batch_ms = search_->GetTimeSinceStart();
      }
      auto nps = search_->GetTotalPlayouts() * 1e3f / time_since_first_batch_ms;
      if (nps > params_.GetNpsLimit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        break;
      }
    }
  }
}

// 1. Initialize internal structures.
void SearchWorker::InitializeIteration(
    std::unique_ptr<NetworkComputation> computation) {
  computation_ = std::move(computation);
  minibatch_.clear();
  minibatch_.reserve(2 * target_minibatch_size_);
}

// 2. Gather minibatch.
void SearchWorker::GatherMinibatch() {
  int minibatch_size = 0;
  int cur_n = 0;
  {
    SharedMutex::Lock lock(search_->nodes_mutex_);
    cur_n = search_->root_node_->GetN();
  }
  int64_t remaining_n =
      latest_time_manager_hints_.GetEstimatedRemainingPlayouts();
  int collisions_left = CalculateCollisionsLeft(
      std::min(static_cast<int64_t>(cur_n), remaining_n), params_);

  number_out_of_order_ = 0;
  int thread_count = search_->thread_count_.load(std::memory_order_acquire);

  while (minibatch_size < target_minibatch_size_ &&
         number_out_of_order_ < max_out_of_order_) {
    if (minibatch_size > 0 && computation_->UsedBatchSize() == 0) return;

    if (thread_count > 1 && minibatch_size > 0 &&
        static_cast<int>(computation_->UsedBatchSize()) >
            params_.GetIdlingMinimumWork() &&
        thread_count - search_->backend_waiting_counter_.load(
                           std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }

    int new_start = static_cast<int>(minibatch_.size());

    PickNodesToExtend(
        std::min({collisions_left, target_minibatch_size_ - minibatch_size,
                  max_out_of_order_ - number_out_of_order_}));

    int non_collisions = 0;
    for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
      auto& picked_node = minibatch_[i];
      if (picked_node.IsCollision()) {
        continue;
      }
      ++non_collisions;
      ++minibatch_size;
    }

    bool needs_wait = false;
    int ppt_start = new_start;
    if (task_workers_ > 0 &&
        non_collisions >= params_.GetMinimumWorkSizeForProcessing()) {
      const int num_tasks = std::clamp(
          non_collisions / params_.GetMinimumWorkPerTaskForProcessing(), 2,
          task_workers_ + 1);
      int per_worker = non_collisions / num_tasks;
      needs_wait = true;
      ResetTasks();
      int found = 0;
      for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
        auto& picked_node = minibatch_[i];
        if (picked_node.IsCollision()) {
          continue;
        }
        ++found;
        if (found == per_worker && picking_tasks_.size() < static_cast<size_t>(num_tasks -1)) {
          picking_tasks_.emplace_back(ppt_start, i + 1);
          task_count_.fetch_add(1, std::memory_order_acq_rel);
          ppt_start = i + 1;
          found = 0;
        }
      }
       if (ppt_start < static_cast<int>(minibatch_.size())) {
          bool remaining_non_collision = false;
          for(int i=ppt_start; i < static_cast<int>(minibatch_.size()); ++i) {
              if (!minibatch_[i].IsCollision()) {
                  remaining_non_collision = true;
                  break;
              }
          }
          if (remaining_non_collision) {
              picking_tasks_.emplace_back(ppt_start, static_cast<int>(minibatch_.size()));
              task_count_.fetch_add(1, std::memory_order_acq_rel);
              ppt_start = static_cast<int>(minibatch_.size());
          }
      }
    }
    ProcessPickedTask(ppt_start, static_cast<int>(minibatch_.size()),
                      &main_workspace_);
    if (needs_wait) {
      WaitForTasks();
    }
    bool some_ooo = false;
    for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
      if (minibatch_[i].ooo_completed) {
        some_ooo = true;
        break;
      }
    }
    if (some_ooo) {
      SharedMutex::Lock lock(search_->nodes_mutex_);
      for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start;
           i--) {
        if (minibatch_[i].IsCollision()) {
          Node* node = minibatch_[i].node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
               if (!node) break;
            node->CancelScoreUpdate(minibatch_[i].multivisit);
          }
          minibatch_.erase(minibatch_.begin() + i);
        } else if (minibatch_[i].ooo_completed) {
          DoBackupUpdateSingleNode(minibatch_[i]);
          minibatch_.erase(minibatch_.begin() + i);
          --minibatch_size;
          ++number_out_of_order_;
        }
      }
    }

    for (size_t i = new_start; i < minibatch_.size(); i++) {
      auto& picked_node = minibatch_[i];
      if (picked_node.IsCollision()) {
        if (picked_node.maxvisit > 0 &&
            collisions_left > picked_node.multivisit) {
          SharedMutex::Lock lock(search_->nodes_mutex_);
          int extra = std::min(picked_node.maxvisit, collisions_left) -
                      picked_node.multivisit;
          picked_node.multivisit += extra;
          Node* node = picked_node.node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
               if (!node) break;
            node->IncrementNInFlight(extra);
          }
        }
        if ((collisions_left -= picked_node.multivisit) <= 0) return;
        if (search_->stop_.load(std::memory_order_acquire)) return;
      }
    }
  }
}

// 2b. Copy collisions into shared collisions.
void SearchWorker::CollectCollisions() {
  SharedMutex::Lock lock(search_->nodes_mutex_);
  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.node,
                                               node_to_process.multivisit);
    }
  }
}

// 3. Prefetch into cache.
void SearchWorker::MaybePrefetchIntoCache() {
  if (search_->stop_.load(std::memory_order_acquire)) return;
  if (computation_->UsedBatchSize() > 0 &&
      static_cast<int>(computation_->UsedBatchSize()) <
          params_.GetMaxPrefetchBatch()) {
    history_.Trim(search_->played_history_.GetLength());
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    PrefetchIntoCache(
        search_->root_node_,
        params_.GetMaxPrefetchBatch() - computation_->UsedBatchSize(), false);
  }
}

// Prefetches up to @budget nodes into cache. Returns number of nodes prefetched.
int SearchWorker::PrefetchIntoCache(Node* node, int budget, bool is_odd_depth) {
  const float draw_score = search_->GetDrawScore(is_odd_depth);
  if (budget <= 0) return 0;

  if (!node || node->GetNStarted() == 0) {
    if (AddNodeToComputation(node)) {
      return 1;
    }
    return 1;
  }
  assert(node);
  if (node->GetN() == 0) return 0;
  if (node->IsTerminal()) return 0;

  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  std::vector<ScoredEdge> scores;
  const float cpuct = ComputeCpuct(params_, node->GetN(), node == search_->root_node_);
  const float puct_mult = cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float fpu = GetFpu(params_, node, node == search_->root_node_, draw_score);
  for (auto& edge : node->Edges()) {
    if (edge.GetP() == 0.0f) continue;
    scores.emplace_back(-edge.GetU(puct_mult) - edge.GetQ(fpu, draw_score),
                        edge);
  }

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;

  for (size_t i = 0; i < scores.size(); ++i) {
    if (search_->stop_.load(std::memory_order_acquire)) break;
    if (budget <= 0) break;

    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index =
          std::min((int)scores.size(), (int)(budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3));
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end(),
                        [](const ScoredEdge& a, const ScoredEdge& b) {
                          return a.first < b.first;
                        });
      first_unsorted_index = new_unsorted_index;
    }

    auto edge = scores[i].second;
    if (i != scores.size() - 1) {
      const float next_score = -scores[i + 1].first;
      const float q = edge.GetQ(-fpu, draw_score);
      if (next_score > q) {
        budget_to_spend =
            std::min(budget, int(edge.GetP() * puct_mult / (next_score - q) -
                                 edge.GetNStarted()) +
                                 1);
      } else {
        budget_to_spend = budget;
      }
    }
    history_.Append(edge.GetMove());
    const int budget_spent =
        PrefetchIntoCache(edge.node(), budget_to_spend, !is_odd_depth);
    history_.Pop();
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}

// Returns whether node was already in cache.
bool SearchWorker::AddNodeToComputation(Node* node) {
  std::vector<Move> moves;
  if (node && node->HasChildren()) {
    moves.reserve(node->GetNumEdges());
    for (const auto& edge : node->Edges()) moves.emplace_back(edge.GetMove());
  } else {
     if (node) {
         moves = history_.Last().GetBoard().GenerateLegalMoves(); // Use history only if node exists
     } else {
         LOGFILE << "Warning: AddNodeToComputation called with null node.";
         return true; // Indicate cached/terminal to avoid NN call
     }
  }
  // Use NetworkComputation::AddInput and Network::Output
  return computation_->AddInput(EvalPosition{history_.GetPositions(), moves},
                                nullptr) == // Pass nullptr for eval result ptr
         NetworkComputation::FETCHED_IMMEDIATELY;
}


// 4. Run NN computation.
void SearchWorker::RunNNComputation() {
  if (computation_->UsedBatchSize() > 0) computation_->ComputeBlocking();
}

// 5. Retrieve NN computations (and terminal values) into nodes.
void SearchWorker::FetchMinibatchResults() {
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process);
  }
}

void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process) {
  if (node_to_process->IsCollision()) return;
  Node* node = node_to_process->node;
  if (!node_to_process->nn_queried) {
    node_to_process->eval->q = node->GetWL();
    node_to_process->eval->d = node->GetD();
    node_to_process->eval->m = node->GetM();
    return;
  }

  // Use NetworkComputation::GetOutput
  auto nn_output = computation_->GetOutput(node_to_process->eval.get());
  if (!nn_output) {
      LOGFILE << "Error: Failed to get NN output for node.";
      // Fallback: Use existing node values
      node_to_process->eval->q = node->GetWL();
      node_to_process->eval->d = node->GetD();
      node_to_process->eval->m = node->GetM();
      // Create empty policy if needed
      if (node->HasChildren()) {
           node_to_process->eval->p.assign(node->GetNumEdges(), 0.0f);
      }
      return;
  }

  // Copy results from nn_output to node_to_process->eval
  node_to_process->eval->q = nn_output->q;
  node_to_process->eval->d = nn_output->d;
  node_to_process->eval->m = nn_output->m;
  node_to_process->eval->p = nn_output->p;

  // Flip value to be from player-to-move perspective.
  node_to_process->eval->q = -node_to_process->eval->q;

  if (params_.GetWDLRescaleRatio() != 1.0f ||
      (params_.GetWDLRescaleDiff() != 0.0f &&
       search_->contempt_mode_ != ContemptMode::NONE)) {
    bool root_stm = (search_->contempt_mode_ == ContemptMode::BLACK) ==
                    search_->played_history_.Last().IsBlackToMove();
    auto sign = (root_stm ^ (node_to_process->depth & 1)) ? 1.0f : -1.0f;
    WDLRescale(node_to_process->eval->q, node_to_process->eval->d,
               params_.GetWDLRescaleRatio(),
               search_->contempt_mode_ == ContemptMode::NONE
                   ? 0
                   : params_.GetWDLRescaleDiff(),
               sign, false, params_.GetWDLMaxS());
  }

  // Add Dirichlet noise if enabled and at root.
  if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
      ApplyDirichletNoise(node, params_.GetNoiseEpsilon(), params_.GetNoiseAlpha());
  }

  // Now policy...
  if (node->HasChildren() && node_to_process->eval->p.size() == node->GetNumEdges()) {
      for (size_t p_idx = 0; auto& edge : node->Edges()) {
          edge.edge()->SetP(node_to_process->eval->p[p_idx++]);
      }
      node->SortEdges();
  } else if (node->HasChildren()) {
       LOGFILE << "Warning: Policy size mismatch during fetch. Node edges: " << (int)node->GetNumEdges()
               << ", Eval policy size: " << node_to_process->eval->p.size();
  } else {
       LOGFILE << "Warning: Fetching results for node with no children/edges.";
  }
}

// 6. Propagate the new nodes' information to all their parents in the tree.
void SearchWorker::DoBackupUpdate() {
  SharedMutex::Lock lock(search_->nodes_mutex_);

  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done) return;
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}

void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  Node* node = node_to_process.node;
  if (node_to_process.IsCollision()) {
    return;
  }

  auto update_parent_bounds =
      params_.GetStickyEndgames() && node->IsTerminal() && !node->GetN();

  float v = node_to_process.eval->q;
  float d = node_to_process.eval->d;
  float m = node_to_process.eval->m;
  int n_to_fix = 0;
  float v_delta = 0.0f;
  float d_delta = 0.0f;
  float m_delta = 0.0f;
  uint32_t solid_threshold =
      static_cast<uint32_t>(params_.GetSolidTreeThreshold());
  for (Node *n = node, *p; n != search_->root_node_->GetParent(); n = p) {
    p = n->GetParent();

    if (n->IsTerminal()) {
      v = n->GetWL();
      d = n->GetD();
      m = n->GetM();
    }
    n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
    if (n_to_fix > 0 && !n->IsTerminal()) {
      n->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
    }
    if (n->GetN() >= solid_threshold) {
      if (n->MakeSolid() && n == search_->root_node_) {
        search_->current_best_edge_ =
            search_->GetBestChildNoTemperature(search_->root_node_, 0);
      }
    }

    if (!p) break;

    bool old_update_parent_bounds = update_parent_bounds;
    if (p->IsTerminal()) n_to_fix = 0;
    update_parent_bounds =
        update_parent_bounds && p != search_->root_node_ && !p->IsTerminal() &&
        MaybeSetBounds(p, m, &n_to_fix, &v_delta, &d_delta, &m_delta);

    v = -v;
    v_delta = -v_delta;
    m++;

    if (p == search_->root_node_ &&
        ((old_update_parent_bounds && n->IsTerminal()) ||
         (n != search_->current_best_edge_.node() &&
          search_->current_best_edge_.GetN() <= n->GetN()))) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_, 0);
    }
  }
  search_->total_playouts_ += node_to_process.multivisit;
  search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
}

bool SearchWorker::MaybeSetBounds(Node* p, float m, int* n_to_fix,
                                  float* v_delta, float* d_delta,
                                  float* m_delta) const {
  auto losing_m = 0.0f;
  auto prefer_tb = false;

  auto lower = GameResult::BLACK_WON;
  auto upper = GameResult::BLACK_WON;
  for (const auto& edge : p->Edges()) {
    const auto [edge_lower, edge_upper] = edge.GetBounds();
    lower = std::max(edge_lower, lower);
    upper = std::max(edge_upper, upper);

    const auto is_tb = edge.IsTbTerminal();
    if (edge_lower == GameResult::WHITE_WON && !is_tb) {
      prefer_tb = false;
      break;
    } else if (edge_upper == GameResult::BLACK_WON) {
      losing_m = std::max(losing_m, edge.GetM(0.0f));
    }
    prefer_tb = prefer_tb || is_tb;
  }

  if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) {
    return false;
  } else if (lower == upper) {
    *n_to_fix = p->GetN();
    assert(*n_to_fix > 0);
    float cur_v = p->GetWL();
    float cur_d = p->GetD();
    float cur_m = p->GetM();
    p->MakeTerminal(
        -upper,
        (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f,
        prefer_tb ? Node::Terminal::Tablebase : Node::Terminal::EndOfGame);
    *v_delta = -(p->GetWL() - cur_v);
    *d_delta = p->GetD() - cur_d;
    *m_delta = p->GetM() - cur_m;
  } else {
    p->SetBounds(-upper, -lower);
  }

  return true;
}

// 7. Update the Search's status and progress information.
void SearchWorker::UpdateCounters() {
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo();

  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

} // No classic namespace
}  // namespace lczero
