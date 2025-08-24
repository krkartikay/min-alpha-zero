import chess
import numpy as np

def tensor_to_board(board_tensor):
    """Convert 7x8x8 tensor back to chess board"""
    board = chess.Board(fen=None)
    board.clear()

    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    # Decode pieces
    for channel in range(6):
        for rank in range(8):
            for file in range(8):
                value = board_tensor[channel, rank, file].item()
                if value != 0:
                    square = chess.square(file, rank)
                    piece_type = piece_types[channel]
                    color = chess.WHITE if value > 0 else chess.BLACK
                    piece = chess.Piece(piece_type, color)
                    board.set_piece_at(square, piece)

    # Set turn
    turn_value = board_tensor[6, 0, 0].item()
    board.turn = chess.WHITE if turn_value > 0 else chess.BLACK

    return board


def action_to_san(action, board):
    """Convert action index to SAN notation"""
    try:
        from_sq = action // 64
        to_sq = action % 64
        from_square = chess.Square(from_sq)
        to_square = chess.Square(to_sq)

        # Try to create a move
        move = chess.Move(from_square, to_square)

        # Check for promotions
        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            if (piece.color == chess.WHITE and to_sq >= 56) or (
                piece.color == chess.BLACK and to_sq <= 7
            ):
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)

        # Check if move is legal
        if move in board.legal_moves:
            return board.san(move)
        else:
            # Try different promotions
            for promotion in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
                move = chess.Move(from_square, to_square, promotion=promotion)
                if move in board.legal_moves:
                    return board.san(move)

            # If still not legal, return algebraic notation
            return chess.square_name(from_sq) + chess.square_name(to_sq)
    except:
        return f"action{action}"

def board_to_tensor(board):
    """Convert chess board to tensor format matching chess_utils.hpp"""
    tensor = np.zeros((7, 8, 8), dtype=np.float32)

    # Convert piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            piece_type = piece.piece_type - 1  # 0-based indexing
            color = 1.0 if piece.color == chess.WHITE else -1.0
            tensor[piece_type, rank, file] = color

    # Add turn channel (channel 6)
    turn_value = 1.0 if board.turn == chess.WHITE else -1.0
    tensor[6, :, :] = turn_value

    return tensor


def move_to_int(move):
    """Convert move to integer format matching chess_utils.hpp"""
    return move.from_square * 64 + move.to_square


def index_to_coordinates(i):
    """Map policy index to coordinates in the 64x64 grid."""
    from_sq = i // 64
    to_sq = i % 64
    x_big = from_sq // 8
    y_big = from_sq % 8
    x_small = to_sq // 8
    y_small = to_sq % 8
    x_all = x_big * 8 + x_small
    y_all = y_big * 8 + y_small
    return x_all, y_all