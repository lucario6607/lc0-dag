/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  ... (License header remains the same) ...
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector> // Use std::vector
#include <list>   // For GCQueue
#include <atomic> // Added for atomic_uint32_t

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/position.h" // Needed for PositionHistory etc.
#include "neural/encoder.h"
#include "proto/net.pb.h"
#include "utils/mutex.h"
#include "neural/network.h" // Include for EvalResult

namespace lczero { // No classic namespace

class Node;
class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  // Debug information about the edge.
  std::string DebugString() const;

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;
  friend class Node;
};

struct Eval {
  float wl;
  float d;
  float ml;
};

// Forward declare EvalResult if needed (depends on neural/network.h)
using EvalResult = neural::EvalResult; // Alias for convenience

class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;

template <bool is_const>
class VisitedNode_Iterator;

typedef std::list<uint64_t> GCQueue; // Defined here, likely from nodetree.h previously
class NodeTree; // Forward declaration

class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  // Takes pointer to a parent node and own index in a parent.
  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(false) {}

  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  Node* CreateSingleChildNode(Move m);
  void CreateEdges(const MoveList& moves);
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const { return n_in_flight_; }
  uint32_t GetChildrenVisits() const { return n_ > 0 ? n_ - 1 : 0; }
  int GetNStarted() const { return n_ + n_in_flight_; }
  float GetQ(float draw_score) const { return wl_ + draw_score * d_; }
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<GameResult, GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();
  void SetBounds(GameResult lower, GameResult upper);
  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(float v, float d, float m, int multivisit);
  void AdjustForTerminal(float v, float d, float m, int multivisit);
  void RevertTerminalVisits(float v, float d, float m, int multivisit);
  void IncrementNInFlight(int multivisit) { n_in_flight_ += multivisit; }
  ConstIterator Edges() const;
  Iterator Edges();
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();
  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node);
  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const;
  std::string DebugString() const;
  bool MakeSolid();
  void SortEdges();
  uint16_t Index() const { return index_; }

  ~Node();

 private:
  void UpdateChildrenParents();

  double wl_ = 0.0f;
  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  float d_ = 0.0f;
  float m_ = 0.0f;
  uint32_t n_ = 0;
  std::atomic_uint32_t n_in_flight_ = 0;
  uint16_t index_;
  uint8_t num_edges_ = 0;
  Terminal terminal_type_ : 2;
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  bool solid_children_ : 1;

  friend class NodeTree;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class Edge;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
};

#if defined(_M_IX86)
#define __i386__
#endif
#if defined(_M_ARM) && !defined(_M_AMD64)
#define __arm__
#endif

#if defined(__i386__) || (defined(__arm__) && !defined(__aarch64__))
static_assert(sizeof(Node) == 48, "Unexpected size of Node for 32bit compile");
#else
static_assert(sizeof(Node) == 64, "Unexpected size of Node");
#endif

class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
  bool operator==(const EdgeAndNode& other) const { return edge_ == other.edge_; }
  bool operator!=(const EdgeAndNode& other) const { return edge_ != other.edge_; }
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  float GetQ(float default_q, float draw_score) const { return (node_ && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q; }
  float GetWL(float default_wl) const { return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl; }
  float GetD(float default_d) const { return (node_ && node_->GetN() > 0) ? node_->GetD() : default_d; }
  float GetM(float default_m) const { return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m; }
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Node::Bounds GetBounds() const { return node_ ? node_->GetBounds() : Node::Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON}; }
  float GetP() const { return edge_->GetP(); }
  Move GetMove(bool flip = false) const { return edge_ ? edge_->GetMove(flip) : Move(); }
  float GetU(float numerator) const { return numerator * GetP() / (1 + GetNStarted()); }
  std::string DebugString() const;

 protected:
  Edge* edge_ = nullptr;
  Node* node_ = nullptr;
};

template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*, std::unique_ptr<Node>*>;
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  Edge_Iterator() {}
  Edge_Iterator(const Node& parent_node, Ptr child_ptr)
      : EdgeAndNode(parent_node.edges_.get(), nullptr),
        node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_) {
    if (edge_ && child_ptr != nullptr) Actualize();
    if (edge_ && child_ptr == nullptr) { node_ = parent_node.child_.get(); }
  }

  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }

  void operator++() {
    if (++current_idx_ == total_count_) { edge_ = nullptr; }
    else { ++edge_; if (node_ptr_ != nullptr) Actualize(); else ++node_; }
  }
  Edge_Iterator& operator*() { return *this; }

  Node* GetOrSpawnNode(Node* parent); // Definition likely in node.cc

 private:
  void Actualize(); // Definition likely in node.cc

  Ptr node_ptr_;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

template <bool is_const>
class VisitedNode_Iterator {
 public:
  VisitedNode_Iterator() {}
  VisitedNode_Iterator(const Node& parent_node, Node* child_ptr)
      : node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_),
        solid_(parent_node.solid_children_) {
    if (node_ptr_ != nullptr && node_ptr_->GetN() == 0) { operator++(); }
  }
  bool operator==(const VisitedNode_Iterator<is_const>& other) const { return node_ptr_ == other.node_ptr_; }
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const { return node_ptr_ != other.node_ptr_; }
  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }
  void operator++(); // Definition likely in node.cc
  Node* operator*(); // Definition likely in node.cc

 private:
  Node* node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
  bool solid_ = false;
};

inline VisitedNode_Iterator<true> Node::VisitedNodes() const { return {*this, child_.get()}; }
inline VisitedNode_Iterator<false> Node::VisitedNodes() { return {*this, child_.get()}; }

// Forward declare GameState if needed by NodeTree
class GameState;

// Define NodeTree class (assuming it should be in a separate file usually)
class NodeTree {
 public:
  // Assume constructor takes SearchParams& or OptionsDict& if needed
  NodeTree(const SearchParams& params);
  NodeTree(); // Default constructor if needed
  ~NodeTree();
  void MakeMove(Move move);
  void TrimTreeAtHead();
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<std::string>& moves);
  bool ResetToPosition(const GameState& pos); // Add this if needed
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  void DeallocateTree();
  Node* current_head_ = nullptr;
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
  // Add member for hash_history_length_ if used
  int hash_history_length_ = 8; // Example default if needed
};


} // namespace lczero/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  ... (License header remains the same) ...
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <mutex>
#include <vector> // Use std::vector
#include <list>   // For GCQueue
#include <atomic> // Added for atomic_uint32_t

#include "chess/board.h"
#include "chess/callbacks.h"
#include "chess/position.h" // Needed for PositionHistory etc.
#include "neural/encoder.h"
#include "proto/net.pb.h"
#include "utils/mutex.h"
#include "neural/network.h" // Include for EvalResult

namespace lczero { // No classic namespace

class Node;
class Edge {
 public:
  // Creates array of edges from the list of moves.
  static std::unique_ptr<Edge[]> FromMovelist(const MoveList& moves);

  // Returns move from the point of view of the player making it (if as_opponent
  // is false) or as opponent (if as_opponent is true).
  Move GetMove(bool as_opponent = false) const;

  // Returns or sets value of Move policy prior returned from the neural net
  // (but can be changed by adding Dirichlet noise). Must be in [0,1].
  float GetP() const;
  void SetP(float val);

  // Debug information about the edge.
  std::string DebugString() const;

 private:
  // Move corresponding to this node. From the point of view of a player,
  // i.e. black's e7e5 is stored as e2e4.
  // Root node contains move a1a1.
  Move move_;

  // Probability that this move will be made, from the policy head of the neural
  // network; compressed to a 16 bit format (5 bits exp, 11 bits significand).
  uint16_t p_ = 0;
  friend class Node;
};

struct Eval {
  float wl;
  float d;
  float ml;
};

// Forward declare EvalResult if needed (depends on neural/network.h)
using EvalResult = neural::EvalResult; // Alias for convenience

class EdgeAndNode;
template <bool is_const>
class Edge_Iterator;

template <bool is_const>
class VisitedNode_Iterator;

typedef std::list<uint64_t> GCQueue; // Defined here, likely from nodetree.h previously
class NodeTree; // Forward declaration

class Node {
 public:
  using Iterator = Edge_Iterator<false>;
  using ConstIterator = Edge_Iterator<true>;

  enum class Terminal : uint8_t { NonTerminal, EndOfGame, Tablebase, TwoFold };

  // Takes pointer to a parent node and own index in a parent.
  Node(Node* parent, uint16_t index)
      : parent_(parent),
        index_(index),
        terminal_type_(Terminal::NonTerminal),
        lower_bound_(GameResult::BLACK_WON),
        upper_bound_(GameResult::WHITE_WON),
        solid_children_(false) {}

  Node(Node&& move_from) = default;
  Node& operator=(Node&& move_from) = default;

  Node* CreateSingleChildNode(Move m);
  void CreateEdges(const MoveList& moves);
  Node* GetParent() const { return parent_; }
  bool HasChildren() const { return static_cast<bool>(edges_); }
  float GetVisitedPolicy() const;
  uint32_t GetN() const { return n_; }
  uint32_t GetNInFlight() const { return n_in_flight_; }
  uint32_t GetChildrenVisits() const { return n_ > 0 ? n_ - 1 : 0; }
  int GetNStarted() const { return n_ + n_in_flight_; }
  float GetQ(float draw_score) const { return wl_ + draw_score * d_; }
  float GetWL() const { return wl_; }
  float GetD() const { return d_; }
  float GetM() const { return m_; }
  bool IsTerminal() const { return terminal_type_ != Terminal::NonTerminal; }
  bool IsTbTerminal() const { return terminal_type_ == Terminal::Tablebase; }
  bool IsTwoFoldTerminal() const { return terminal_type_ == Terminal::TwoFold; }
  typedef std::pair<GameResult, GameResult> Bounds;
  Bounds GetBounds() const { return {lower_bound_, upper_bound_}; }
  uint8_t GetNumEdges() const { return num_edges_; }
  void CopyPolicy(int max_needed, float* output) const;
  void MakeTerminal(GameResult result, float plies_left = 0.0f,
                    Terminal type = Terminal::EndOfGame);
  void MakeNotTerminal();
  void SetBounds(GameResult lower, GameResult upper);
  bool TryStartScoreUpdate();
  void CancelScoreUpdate(int multivisit);
  void FinalizeScoreUpdate(float v, float d, float m, int multivisit);
  void AdjustForTerminal(float v, float d, float m, int multivisit);
  void RevertTerminalVisits(float v, float d, float m, int multivisit);
  void IncrementNInFlight(int multivisit) { n_in_flight_ += multivisit; }
  ConstIterator Edges() const;
  Iterator Edges();
  VisitedNode_Iterator<true> VisitedNodes() const;
  VisitedNode_Iterator<false> VisitedNodes();
  void ReleaseChildren();
  void ReleaseChildrenExceptOne(Node* node);
  Edge* GetEdgeToNode(const Node* node) const;
  Edge* GetOwnEdge() const;
  std::string DebugString() const;
  bool MakeSolid();
  void SortEdges();
  uint16_t Index() const { return index_; }

  ~Node();

 private:
  void UpdateChildrenParents();

  double wl_ = 0.0f;
  std::unique_ptr<Edge[]> edges_;
  Node* parent_ = nullptr;
  std::unique_ptr<Node> child_;
  std::unique_ptr<Node> sibling_;
  float d_ = 0.0f;
  float m_ = 0.0f;
  uint32_t n_ = 0;
  std::atomic_uint32_t n_in_flight_ = 0;
  uint16_t index_;
  uint8_t num_edges_ = 0;
  Terminal terminal_type_ : 2;
  GameResult lower_bound_ : 2;
  GameResult upper_bound_ : 2;
  bool solid_children_ : 1;

  friend class NodeTree;
  friend class Edge_Iterator<true>;
  friend class Edge_Iterator<false>;
  friend class Edge;
  friend class VisitedNode_Iterator<true>;
  friend class VisitedNode_Iterator<false>;
};

#if defined(_M_IX86)
#define __i386__
#endif
#if defined(_M_ARM) && !defined(_M_AMD64)
#define __arm__
#endif

#if defined(__i386__) || (defined(__arm__) && !defined(__aarch64__))
static_assert(sizeof(Node) == 48, "Unexpected size of Node for 32bit compile");
#else
static_assert(sizeof(Node) == 64, "Unexpected size of Node");
#endif

class EdgeAndNode {
 public:
  EdgeAndNode() = default;
  EdgeAndNode(Edge* edge, Node* node) : edge_(edge), node_(node) {}
  void Reset() { edge_ = nullptr; }
  explicit operator bool() const { return edge_ != nullptr; }
  bool operator==(const EdgeAndNode& other) const { return edge_ == other.edge_; }
  bool operator!=(const EdgeAndNode& other) const { return edge_ != other.edge_; }
  bool HasNode() const { return node_ != nullptr; }
  Edge* edge() const { return edge_; }
  Node* node() const { return node_; }

  float GetQ(float default_q, float draw_score) const { return (node_ && node_->GetN() > 0) ? node_->GetQ(draw_score) : default_q; }
  float GetWL(float default_wl) const { return (node_ && node_->GetN() > 0) ? node_->GetWL() : default_wl; }
  float GetD(float default_d) const { return (node_ && node_->GetN() > 0) ? node_->GetD() : default_d; }
  float GetM(float default_m) const { return (node_ && node_->GetN() > 0) ? node_->GetM() : default_m; }
  uint32_t GetN() const { return node_ ? node_->GetN() : 0; }
  int GetNStarted() const { return node_ ? node_->GetNStarted() : 0; }
  uint32_t GetNInFlight() const { return node_ ? node_->GetNInFlight() : 0; }
  bool IsTerminal() const { return node_ ? node_->IsTerminal() : false; }
  bool IsTbTerminal() const { return node_ ? node_->IsTbTerminal() : false; }
  Node::Bounds GetBounds() const { return node_ ? node_->GetBounds() : Node::Bounds{GameResult::BLACK_WON, GameResult::WHITE_WON}; }
  float GetP() const { return edge_->GetP(); }
  Move GetMove(bool flip = false) const { return edge_ ? edge_->GetMove(flip) : Move(); }
  float GetU(float numerator) const { return numerator * GetP() / (1 + GetNStarted()); }
  std::string DebugString() const;

 protected:
  Edge* edge_ = nullptr;
  Node* node_ = nullptr;
};

template <bool is_const>
class Edge_Iterator : public EdgeAndNode {
 public:
  using Ptr = std::conditional_t<is_const, const std::unique_ptr<Node>*, std::unique_ptr<Node>*>;
  using value_type = Edge_Iterator;
  using iterator_category = std::forward_iterator_tag;
  using difference_type = std::ptrdiff_t;
  using pointer = Edge_Iterator*;
  using reference = Edge_Iterator&;

  Edge_Iterator() {}
  Edge_Iterator(const Node& parent_node, Ptr child_ptr)
      : EdgeAndNode(parent_node.edges_.get(), nullptr),
        node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_) {
    if (edge_ && child_ptr != nullptr) Actualize();
    if (edge_ && child_ptr == nullptr) { node_ = parent_node.child_.get(); }
  }

  Edge_Iterator<is_const> begin() { return *this; }
  Edge_Iterator<is_const> end() { return {}; }

  void operator++() {
    if (++current_idx_ == total_count_) { edge_ = nullptr; }
    else { ++edge_; if (node_ptr_ != nullptr) Actualize(); else ++node_; }
  }
  Edge_Iterator& operator*() { return *this; }

  Node* GetOrSpawnNode(Node* parent); // Definition likely in node.cc

 private:
  void Actualize(); // Definition likely in node.cc

  Ptr node_ptr_;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
};

template <bool is_const>
class VisitedNode_Iterator {
 public:
  VisitedNode_Iterator() {}
  VisitedNode_Iterator(const Node& parent_node, Node* child_ptr)
      : node_ptr_(child_ptr),
        total_count_(parent_node.num_edges_),
        solid_(parent_node.solid_children_) {
    if (node_ptr_ != nullptr && node_ptr_->GetN() == 0) { operator++(); }
  }
  bool operator==(const VisitedNode_Iterator<is_const>& other) const { return node_ptr_ == other.node_ptr_; }
  bool operator!=(const VisitedNode_Iterator<is_const>& other) const { return node_ptr_ != other.node_ptr_; }
  VisitedNode_Iterator<is_const> begin() { return *this; }
  VisitedNode_Iterator<is_const> end() { return {}; }
  void operator++(); // Definition likely in node.cc
  Node* operator*(); // Definition likely in node.cc

 private:
  Node* node_ptr_ = nullptr;
  uint16_t current_idx_ = 0;
  uint16_t total_count_ = 0;
  bool solid_ = false;
};

inline VisitedNode_Iterator<true> Node::VisitedNodes() const { return {*this, child_.get()}; }
inline VisitedNode_Iterator<false> Node::VisitedNodes() { return {*this, child_.get()}; }

// Forward declare GameState if needed by NodeTree
class GameState;

// Define NodeTree class (assuming it should be in a separate file usually)
class NodeTree {
 public:
  // Assume constructor takes SearchParams& or OptionsDict& if needed
  NodeTree(const SearchParams& params);
  NodeTree(); // Default constructor if needed
  ~NodeTree();
  void MakeMove(Move move);
  void TrimTreeAtHead();
  bool ResetToPosition(const std::string& starting_fen,
                       const std::vector<std::string>& moves);
  bool ResetToPosition(const GameState& pos); // Add this if needed
  const Position& HeadPosition() const { return history_.Last(); }
  int GetPlyCount() const { return HeadPosition().GetGamePly(); }
  bool IsBlackToMove() const { return HeadPosition().IsBlackToMove(); }
  Node* GetCurrentHead() const { return current_head_; }
  Node* GetGameBeginNode() const { return gamebegin_node_.get(); }
  const PositionHistory& GetPositionHistory() const { return history_; }

 private:
  void DeallocateTree();
  Node* current_head_ = nullptr;
  std::unique_ptr<Node> gamebegin_node_;
  PositionHistory history_;
  // Add member for hash_history_length_ if used
  int hash_history_length_ = 8; // Example default if needed
};


} // namespace lczero
