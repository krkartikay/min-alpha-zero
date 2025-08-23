import chess


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
