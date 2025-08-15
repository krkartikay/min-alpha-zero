#include "alpha_zero.h"

const std::string kBeforeCastle =
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";

int main() {
  chess::Board board(kBeforeCastle);
  // gen moves
  chess::Movelist m;
  chess::movegen::legalmoves(m, board);
  for (int i = 0; i < m.size(); ++i) {
    std::string move_san = chess::uci::moveToSan(board, m[i]);
    std::string move_uci = chess::uci::moveToUci(m[i]);
    std::string move_lan = chess::uci::moveToLan(board, m[i]);
    chess::Square move_from = m[i].from();
    chess::Square move_to = m[i].to();
    std::cout << absl::StrFormat(
        "Move %d: \tSAN: %s \tUCI: (%s) \tLAN: %s\tFrom: %s (%d)\tTo: %s "
        "(%d)\n",
        i, move_san, move_uci, move_lan, move_from, move_from.index(), move_to,
        move_to.index());
  }
}