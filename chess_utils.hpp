#include <array>
#include <chess.hpp>

#ifndef CHESS_UTILS_HPP
#define CHESS_UTILS_HPP

namespace chess {

inline std::array<float, 7 * 8 * 8> board_to_tensor(const chess::Board& board) {
  std::array<float, 7 * 8 * 8> tensor{};
  tensor.fill(0.0f);

  for (int sq = 0; sq < 64; ++sq) {
    auto piece = board.at(chess::Square(sq));
    if (piece != chess::Piece::NONE) {
      int file = sq % 8;
      int rank = sq / 8;
      int piece_type = static_cast<int>(piece.type());
      int color = static_cast<int>(piece.color());

      int channel = piece_type;
      int idx = channel * 64 + rank * 8 + file;
      tensor[idx] = (color == 0) ? 1.0f : -1.0f;
    }
  }

  float turn_value = (board.sideToMove() == chess::Color::WHITE) ? 1.0f : -1.0f;
  for (int i = 0; i < 64; ++i) {
    tensor[6 * 64 + i] = turn_value;
  }

  return tensor;
}

inline int move_to_int(const chess::Move& move) {
  int from = static_cast<int>(move.from().index());
  int to = static_cast<int>(move.to().index());
  return from * 64 + to;
}

inline chess::Move int_to_move(int move_int, const chess::Board& board) {
  int from = move_int / 64;
  int to = move_int % 64;
  auto piece = board.at(chess::Square(from));
  // Detect pawn promotion moves: Pawn moving to the last rank
  if (((from / 8 == 6 && to / 8 == 7) ||
      (from / 8 == 1 && to / 8 == 0) )&&
          (piece.type() == chess::PieceType::PAWN)) {
    return chess::Move::make<chess::Move::PROMOTION>(
        chess::Square(from), chess::Square(to),
        chess::PieceType::QUEEN);  // Default promotion to queen
  }
  // Detect castling: king moving to rook squares
  if (piece.type() == chess::PieceType::KING &&
      ((from == 4 && (to == 0 || to == 7)) ||     // White castling
       (from == 60 && (to == 56 || to == 63)))) { // Black castling
    return chess::Move::make<chess::Move::CASTLING>(chess::Square(from),
                                                    chess::Square(to));
  }
  return chess::Move::make(chess::Square(from), chess::Square(to));
}

}  // namespace chess

#endif  // CHESS_UTILS_HPP