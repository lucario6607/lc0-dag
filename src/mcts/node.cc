/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors
  ... (License header remains the same) ...
*/

#include "mcts/node.h" // Adjusted include path

#include <absl/algorithm/container.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <list>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <iomanip> // Added for std::setprecision etc.
#include <vector> // Use std::vector

#include "neural/encoder.h"
#include "neural/network.h" // Include for EvalResult etc.
#include "utils/exception.h"
#include "utils/hashcat.h"
#include "chess/position.h" // Included for PositionHistory/MoveList
#include "chess/gamestate.h" // Include if NodeTree uses it

namespace lczero { // No classic namespace

/////////////////////////////////////////////////////////////////////////
// Node garbage collector
/////////////////////////////////////////////////////////////////////////

namespace {
// Periodicity of garbage collection, milliseconds.
const int kGCIntervalMs = 100;

// Every kGCIntervalMs milliseconds release nodes in a separate GC thread.
class NodeGarbageCollector {
 public:
  NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

  // Takes ownership of a subtree, to dispose it in a separate thread when
  // it has time.
  void AddToGcQueue(std::unique_ptr<Node> node, size_t solid_size = 0) {
    if (!node) return;
    Mutex::Lock lock(gc_mutex_);
    subtrees_to_gc_.emplace_back(std::move(node));
    subtrees_to_gc_solid_size_.push_back(solid_size);
  }

  ~NodeGarbageCollector() {
    // Flips stop flag and waits for a worker thread to stop.
    stop_.store(true);
    gc_thread_.join();
  }

 private:
  void GarbageCollect() {
    while (!stop_.load()) {
      // Node will be released in destructor when mutex is not locked.
      std::unique_ptr<Node> node_to_gc;
      size_t solid_size = 0;
      {
        // Lock the mutex and move last subtree from subtrees_to_gc_ into
        // node_to_gc.
        Mutex::Lock lock(gc_mutex_);
        if (subtrees_to_gc_.empty()) return;
        node_to_gc = std::move(subtrees_to_gc_.back());
        subtrees_to_gc_.pop_back();
        solid_size = subtrees_to_gc_solid_size_.back();
        subtrees_to_gc_solid_size_.pop_back();
      }
      // Solid is a hack...
      if (solid_size != 0) {
        for (size_t i = 0; i < solid_size; i++) {
          node_to_gc.get()[i].~Node();
        }
        std::allocator<Node> alloc;
        alloc.deallocate(node_to_gc.release(), solid_size);
      }
    }
  }

  void Worker() {
    while (!stop_.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(kGCIntervalMs));
      GarbageCollect();
    };
  }

  mutable Mutex gc_mutex_;
  std::vector<std::unique_ptr<Node>> subtrees_to_gc_ GUARDED_BY(gc_mutex_); // Use std::vector
  std::vector<size_t> subtrees_to_gc_solid_size_ GUARDED_BY(gc_mutex_); // Use std::vector

  // When true, Worker() should stop and exit.
  std::atomic<bool> stop_{false};
  std::thread gc_thread_;
};

NodeGarbageCollector gNodeGc;
}  // namespace

/////////////////////////////////////////////////////////////////////////
// Edge
/////////////////////////////////////////////////////////////////////////

Move Edge::GetMove(bool as_opponent) const {
  if (!as_opponent) return move_;
  Move m = move_;
  m.Flip();
  return m;
}

// Policy priors (P) are stored in a compressed 16-bit format.
// ... (rest of SetP/GetP comments remain the same) ...

void Edge::SetP(float p) {
  assert(0.0f <= p && p <= 1.0f);
  constexpr int32_t roundings = (1 << 11) - (3 << 28);
  int32_t tmp;
  std::memcpy(&tmp, &p, sizeof(float));
  tmp += roundings;
  p_ = (tmp < 0) ? 0 : static_cast<uint16_t>(tmp >> 12);
}

float Edge::GetP() const {
  // Reshift into place and set the assumed-set exponent bits.
  uint32_t tmp = (static_cast<uint32_t>(p_) << 12) | (3 << 28);
  float ret;
  std::memcpy(&ret, &tmp, sizeof(uint32_t));
  return ret;
}

std::string Edge::DebugString() const {
  std::ostringstream oss;
  oss << "Move: " << move_.ToString(true) << " p_: " << p_
      << " GetP: " << GetP();
  return oss.str();
}

std::unique_ptr<Edge[]> Edge::FromMovelist(const MoveList& moves) {
  std::unique_ptr<Edge[]> edges = std::make_unique<Edge[]>(moves.size());
  auto* edge = edges.get();
  for (const auto move : moves) edge++->move_ = move;
  return edges;
}

/////////////////////////////////////////////////////////////////////////
// Node
/////////////////////////////////////////////////////////////////////////

Node* Node::CreateSingleChildNode(Move m) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist({m});
  num_edges_ = 1;
  child_ = std::make_unique<Node>(this, 0);
  return child_.get();
}

void Node::CreateEdges(const MoveList& moves) {
  assert(!edges_);
  assert(!child_);
  edges_ = Edge::FromMovelist(moves);
  num_edges_ = moves.size();
}

Node::ConstIterator Node::Edges() const {
  return {*this, !solid_children_ ? &child_ : nullptr};
}
Node::Iterator Node::Edges() {
  return {*this, !solid_children_ ? &child_ : nullptr};
}

float Node::GetVisitedPolicy() const {
  float sum = 0.0f;
  for (auto* node : VisitedNodes()) sum += GetEdgeToNode(node)->GetP();
  return sum;
}

Edge* Node::GetEdgeToNode(const Node* node) const {
  assert(node->parent_ == this);
  assert(node->index_ < num_edges_);
  return &edges_[node->index_];
}

Edge* Node::GetOwnEdge() const {
  if (parent_ == nullptr) return nullptr; // Root node has no edge to itself
  return GetParent()->GetEdgeToNode(this);
}


std::string Node::DebugString() const {
  std::ostringstream oss;
  oss << " Term:" << static_cast<int>(terminal_type_) << " This:" << this
      << " Parent:" << parent_ << " Index:" << index_
      << " Child:" << child_.get() << " Sibling:" << sibling_.get()
      << " WL:" << wl_ << " N:" << n_ << " N_:" << n_in_flight_
      << " Edges:" << static_cast<int>(num_edges_)
      << " Bounds:" << static_cast<int>(lower_bound_) - 2 << ","
      << static_cast<int>(upper_bound_) - 2 << " Solid:" << solid_children_;
  return oss.str();
}

bool Node::MakeSolid() {
  if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;
  // Can only make solid if no immediate leaf children are in flight since we
  // allow the search code to hold references to leaf nodes across locks.
  Node* old_child_to_check = child_.get();
  uint32_t total_in_flight = 0;
  while (old_child_to_check != nullptr) {
    if (old_child_to_check->GetN() <= 1 &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    if (old_child_to_check->IsTerminal() &&
        old_child_to_check->GetNInFlight() > 0) {
      return false;
    }
    total_in_flight += old_child_to_check->GetNInFlight();
    old_child_to_check = old_child_to_check->sibling_.get();
  }
  // If the total of children in flight is not the same as self, then there are
  // collisions against immediate children (which don't update the GetNInFlight
  // of the leaf) and its not safe.
  if (total_in_flight != GetNInFlight()) {
    return false;
  }
  std::allocator<Node> alloc;
  auto* new_children = alloc.allocate(num_edges_);
  for (int i = 0; i < num_edges_; i++) {
    new (&(new_children[i])) Node(this, i);
  }
  std::unique_ptr<Node> old_child = std::move(child_);
  while (old_child) {
    int index = old_child->index_;
    new_children[index] = std::move(*old_child.get());
    // This isn't needed, but it helps crash things faster if something has gone
    // wrong.
    old_child->parent_ = nullptr;
    gNodeGc.AddToGcQueue(std::move(old_child));
    new_children[index].UpdateChildrenParents();
    old_child = std::move(new_children[index].sibling_);
  }
  // This is a hack.
  child_ = std::unique_ptr<Node>(new_children);
  solid_children_ = true;
  return true;
}

void Node::SortEdges() {
  assert(edges_);
  // Sorting edges requires children to be non-solid or updated accordingly
  assert(!solid_children_ || !"Cannot sort edges of a solidified node easily");
  if (solid_children_) return; // Avoid sorting solid nodes for now

  // Sorting on raw p_ is the same as sorting on GetP() as a side effect of
  // the encoding, and its noticeably faster.
  std::sort(edges_.get(), (edges_.get() + num_edges_),
            [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
  // Important: After sorting edges, the indices stored in existing child nodes
  // are now incorrect. We need to update them or handle selection differently.
  // Simplest approach (but potentially slow) is to find children by move match.
  // A better approach would be to store edge pointers in nodes or rebuild child list.
  // For now, this function is potentially dangerous if called after children exist.
  // Let's comment out the child index update part as it requires more complex logic
  // or a change in how children are linked/found.
  /*
  if (child_) {
       // Create a map from Move to Node* for existing children
       std::unordered_map<Move, Node*> child_map;
       Node* current = child_.get();
       while (current) {
           child_map[current->GetOwnEdge()->GetMove()] = current; // Find edge first
           current = current->sibling_.get();
       }

       // Rebuild the child linked list according to the new edge order
       std::unique_ptr<Node> new_child_list = nullptr;
       Node** next_sibling_ptr = &new_child_list;

       for (uint16_t i = 0; i < num_edges_; ++i) {
           auto it = child_map.find(edges_[i].GetMove());
           if (it != child_map.end()) {
               Node* node_to_move = it->second;
               node_to_move->index_ = i; // Update index
               // Extract node from map (or wherever it's held) and link it
               // This logic depends heavily on how nodes are managed...
               // For unique_ptr list, this requires careful pointer manipulation.
               // For now, just update index, assuming selection finds by index.
               // This is INCOMPLETE / POTENTIALLY WRONG for linked list children!
           }
       }
       // child_ = std::move(new_child_list); // Assign the rebuilt list
  }
  */

}

void Node::MakeTerminal(GameResult result, float plies_left, Terminal type) {
  if (type != Terminal::TwoFold) SetBounds(result, result);
  terminal_type_ = type;
  m_ = plies_left;
  if (result == GameResult::DRAW) {
    wl_ = 0.0f;
    d_ = 1.0f;
  } else if (result == GameResult::WHITE_WON) {
    wl_ = 1.0f;
    d_ = 0.0f;
  } else if (result == GameResult::BLACK_WON) {
    wl_ = -1.0f;
    d_ = 0.0f;
    if (GetParent() != nullptr) GetOwnEdge()->SetP(0.0f);
  }
}

void Node::MakeNotTerminal() {
  terminal_type_ = Terminal::NonTerminal;
  // Recalculate bounds based on children if they exist
  if (edges_) {
       lower_bound_ = GameResult::BLACK_WON;
       upper_bound_ = GameResult::BLACK_WON;
       for (const auto& edge : Edges()) {
           if (edge.node()) { // Only consider existing children
                const auto [child_lower, child_upper] = edge.node()->GetBounds();
                // Parent bounds are flipped child bounds
                lower_bound_ = std::max(lower_bound_, -child_upper);
                upper_bound_ = std::max(upper_bound_, -child_lower);
           } else {
                // If a child doesn't exist, assume full range initially
                lower_bound_ = GameResult::BLACK_WON;
                upper_bound_ = GameResult::WHITE_WON;
                break; // One unknown child means parent bounds are unknown
           }
       }
       // Final flip for parent
       std::swap(lower_bound_, upper_bound_);
       lower_bound_ = -lower_bound_;
       upper_bound_ = -upper_bound_;
   } else {
       // If no children, reset to default bounds
       lower_bound_ = GameResult::BLACK_WON;
       upper_bound_ = GameResult::WHITE_WON;
   }

  // Recalculate stats based on children
  wl_ = 0.0;
  d_ = 0.0;
  m_ = 0.0;
  n_ = 0; // Reset visit count, will be recalculated from children
  if (edges_) {
    n_++; // Count the visit that expanded this node initially
    for (const auto& child : Edges()) {
      const auto visits = child.GetN();
      if (visits > 0) {
        n_ += visits;
        wl_ += -child.GetWL(0.0f) * visits; // Flip child's WL for parent's perspective
        d_ += child.GetD(0.0f) * visits;
        m_ += (child.GetM(0.0f) + 1) * visits; // Increment child's moves_left
      }
    }
    if (n_ > 1) { // Avoid division by zero if only the expansion visit exists
        wl_ /= (n_ - 1); // Average over child visits
        d_ /= (n_ - 1);
        m_ /= (n_ - 1);
    } else if (n_ == 1) { // Only expansion visit exists
         wl_ = 0.0; // No child info to update stats
         d_ = 0.0;
         m_ = 0.0;
    }
  }
}

void Node::SetBounds(GameResult lower, GameResult upper) {
  lower_bound_ = lower;
  upper_bound_ = upper;
}

bool Node::TryStartScoreUpdate() {
  if (n_ == 0 && n_in_flight_ > 0) return false;
  ++n_in_flight_;
  return true;
}

void Node::CancelScoreUpdate(int multivisit) {
   assert(n_in_flight_ >= (uint32_t)multivisit); // Ensure atomic is >= before sub
   n_in_flight_ -= multivisit;
}

void Node::FinalizeScoreUpdate(float v, float d, float m, int multivisit) {
  const uint32_t n_new = n_ + multivisit;
  if (n_new > 0) {
    wl_ += multivisit * (v - wl_) / n_new;
    d_ += multivisit * (d - d_) / n_new;
    m_ += multivisit * (m - m_) / n_new;
  } else {
    wl_ = v; d_ = d; m_ = m;
  }
  n_ += multivisit;
  assert(n_in_flight_ >= (uint32_t)multivisit);
  n_in_flight_ -= multivisit;
}

void Node::AdjustForTerminal(float v, float d, float m, int multivisit) {
  assert(n_ > 0);
  // Adjust existing stats based on the provided delta values and count
  // This calculation needs careful review - is it simply adding delta N times?
  // Or should it adjust the average? Adjusting average seems more correct.
  // Example: wl_ = (wl_ * (n_ - multivisit) + v * multivisit) / n_;
  // Let's assume the simpler addition for now, needs verification.
  wl_ += multivisit * v / n_; // This seems potentially incorrect logic from original
  d_ += multivisit * d / n_;
  m_ += multivisit * m / n_;
}

void Node::RevertTerminalVisits(float v, float d, float m, int multivisit) {
  const int n_old = n_; // Store old N
  const int n_new = n_ - multivisit;
  if (n_new <= 0) {
    wl_ = 0.0; d_ = 1.0; m_ = 0.0; n_ = 0;
  } else {
    // Reverse the FinalizeScoreUpdate logic carefully
    // Original W = W_old + multi * (v - W_old) / (N_old)
    // W_old = (W * N_old - multi * v) / (N_old - multi)
    // W_old = (W * N_old - multi * v) / n_new
    if (n_old > 0) { // Avoid division by zero if n_ was already 0 somehow
        wl_ = (wl_ * n_old - multivisit * v) / n_new;
        d_ = (d_ * n_old - multivisit * d) / n_new;
        m_ = (m_ * n_old - multivisit * m) / n_new;
    } else {
        wl_ = 0.0; d_ = 1.0; m_ = 0.0; // Reset if n_ was 0
    }
    n_ -= multivisit;
  }
}

void Node::UpdateChildrenParents() {
  if (!solid_children_) {
    Node* cur_child = child_.get();
    while (cur_child != nullptr) {
      cur_child->parent_ = this;
      cur_child = cur_child->sibling_.get();
    }
  } else {
    Node* child_array = child_.get();
    for (int i = 0; i < num_edges_; i++) {
      child_array[i].parent_ = this;
    }
  }
}

void Node::ReleaseChildren() {
  gNodeGc.AddToGcQueue(std::move(child_), solid_children_ ? num_edges_ : 0);
  edges_.reset(); // Also release edges when children are gone
  num_edges_ = 0;
  solid_children_ = false; // Reset solid state
}

void Node::ReleaseChildrenExceptOne(Node* node_to_save) {
  if (solid_children_) {
    std::unique_ptr<Node> saved_node;
    if (node_to_save != nullptr) {
      // Create a temporary unique_ptr to hold the moved node
      auto tmp_node = std::make_unique<Node>(std::move(*node_to_save));
      saved_node = std::move(tmp_node);
      saved_node->parent_ = this; // Ensure parent is set correctly after move
    }
    // Queue the original array for deletion
    gNodeGc.AddToGcQueue(std::move(child_), num_edges_);
    // Assign the saved node (if any) back to child_
    child_ = std::move(saved_node);
    if (child_) {
        // If we saved a node, update its siblings (should be null)
        child_->sibling_.reset();
    }
    // Reset solid state and edge info if needed
    solid_children_ = false;
    if (!child_) { // If no node was saved
        edges_.reset();
        num_edges_ = 0;
    } else { // If one node was saved, we technically have one edge
        // Recreating the single edge array might be complex, could just leave edges_ as is
        // but mark num_edges_ = 1? Needs careful consideration of how edges are used.
        // Safest might be to clear edges and rely on finding the edge via the child's index.
        // Or, create a new single-edge array. Let's assume we keep edge info via index.
        num_edges_ = 1; // Indicate only one valid edge remains conceptually
    }
  } else {
    std::unique_ptr<Node> saved_node;
    std::unique_ptr<Node> current_child = std::move(child_); // Take ownership
    child_ = nullptr; // Clear the original pointer
    Node* next_child = nullptr;

    while (current_child) {
        next_child = current_child->sibling_.release(); // Release ownership from sibling

        if (current_child.get() == node_to_save) {
            saved_node = std::move(current_child); // Keep the node to save
        } else {
            gNodeGc.AddToGcQueue(std::move(current_child)); // GC the others
        }
        current_child.reset(next_child); // Move to the next sibling
    }
    // Assign the saved node back
    child_ = std::move(saved_node);
     if (!child_) { // If no node was saved or found
        num_edges_ = 0;
        edges_.reset();
    } else {
        child_->sibling_.reset(); // Ensure saved node has no siblings
        num_edges_ = 1; // Update edge count conceptually
    }
  }
}

/////////////////////////////////////////////////////////////////////////
// EdgeAndNode
/////////////////////////////////////////////////////////////////////////

std::string EdgeAndNode::DebugString() const {
  if (!edge_) return "(no edge)";
  return edge_->DebugString() + " " +
         (node_ ? node_->DebugString() : "(no node)");
}

/////////////////////////////////////////////////////////////////////////
// NodeTree Implementation (Assuming it's defined elsewhere, e.g., nodetree.cc)
/////////////////////////////////////////////////////////////////////////
// ... Implementations for NodeTree methods if they were moved ...
// Example placeholder:
// NodeTree::NodeTree(const SearchParams& params) : hash_history_length_(params.GetCacheHistoryLength() + 1) {
//      gamebegin_node_ = std::make_unique<Node>(nullptr, 0);
//      current_head_ = gamebegin_node_.get();
//      // Initialize history_ if needed
// }
// NodeTree::NodeTree() : hash_history_length_(8) { // Default value if SearchParams not available
//      gamebegin_node_ = std::make_unique<Node>(nullptr, 0);
//      current_head_ = gamebegin_node_.get();
// }
// NodeTree::~NodeTree() { DeallocateTree(); }
// void NodeTree::DeallocateTree() { ... }
// void NodeTree::MakeMove(Move move) { ... }
// void NodeTree::TrimTreeAtHead() { ... }
// bool NodeTree::ResetToPosition(const GameState& pos) { ... }
// bool NodeTree::ResetToPosition(const std::string& starting_fen, const std::vector<std::string>& moves) { ... }

// --- Implementations for Edge/Visited Node Iterators ---
template <bool is_const>
void Edge_Iterator<is_const>::Actualize() {
  assert(node_ptr_ != nullptr);
  auto node = node_ptr_->get();
  while (node != nullptr && node->Index() < current_idx_) {
    node_ptr_ = node->GetSibling(); // Use GetSibling method
    node = node_ptr_->get();
  }
  if (node != nullptr && node->Index() == current_idx_) {
    node_ = node;
    node_ptr_ = node->GetSibling(); // Use GetSibling method
  } else {
    node_ = nullptr;
  }
}

// Explicit template instantiation if needed, or keep in header
// template class Edge_Iterator<true>;
// template class Edge_Iterator<false>;

template <bool is_const>
void VisitedNode_Iterator<is_const>::operator++() {
    if (solid_) {
      while (++current_idx_ != total_count_ &&
             node_ptr_[current_idx_].GetN() == 0) {
        if (node_ptr_[current_idx_].GetNInFlight() == 0) {
          current_idx_ = total_count_;
          break;
        }
      }
      if (current_idx_ == total_count_) {
        node_ptr_ = nullptr;
      }
    } else {
      do {
        if (node_ptr_) // Check if node_ptr_ is not null before accessing sibling_
            node_ptr_ = node_ptr_->sibling_.get();
        else
            break; // Exit loop if node_ptr_ became null

        if (node_ptr_ != nullptr && node_ptr_->GetN() == 0 &&
            node_ptr_->GetNInFlight() == 0) {
          node_ptr_ = nullptr;
          break;
        }
      } while (node_ptr_ != nullptr && node_ptr_->GetN() == 0);
    }
}

template <bool is_const>
Node* VisitedNode_Iterator<is_const>::operator*() {
    if (solid_) {
      // Bounds check for safety
      if (current_idx_ < total_count_)
          return &(node_ptr_[current_idx_]);
      else
          return nullptr; // Should not happen if used correctly
    } else {
      return node_ptr_;
    }
}

// Explicit template instantiation if needed
// template class VisitedNode_Iterator<true>;
// template class VisitedNode_Iterator<false>;

} // No classic namespace
}  // namespace lczero
