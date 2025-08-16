import dash
from dash import dcc, html, Input, Output
import chess
import chess.svg
import torch
import base64
import plotly.graph_objects as go
from dataset import TrainingDataset

BOARD_SIZE = 300
IMG_SIZE = 300
CELL_WIDTH = 300
CELL_HEIGHT = 320
TOP_ROW_CELL_HEIGHT = 50  # New constant for top row cell height

# Load the dataset
dataset = TrainingDataset("training_data.bin")

# Initialize the Dash app
app = dash.Dash(__name__)
app.title = "Chess Dataset Explorer"

# Layout of the dashboard
app.layout = html.Table(
    [
        html.Tr(
            [
                html.Td(
                    html.Div(
                        [
                            html.H3("Value"),
                            html.Pre(id="value"),
                        ]
                    ),
                    style={
                        "width": f"{CELL_WIDTH}px",
                        "height": f"{TOP_ROW_CELL_HEIGHT}px",
                    },
                ),
                html.Td(
                    html.Div(
                        [
                            html.H3("Final Value"),
                            html.Pre(id="final-value"),
                        ]
                    ),
                    style={
                        "width": f"{CELL_WIDTH}px",
                        "height": f"{TOP_ROW_CELL_HEIGHT}px",
                    },
                ),
                html.Td(
                    html.Div(
                        [
                            html.Label("Select Record Index:"),
                            dcc.Input(
                                id="record-index",
                                type="number",
                                value=0,
                                min=0,
                                max=len(dataset) - 1,
                            ),
                            html.Button("Previous", id="prev-button"),
                            html.Button("Next", id="next-button"),
                        ]
                    ),
                    style={
                        "width": f"{CELL_WIDTH}px",
                        "height": f"{TOP_ROW_CELL_HEIGHT}px",
                    },
                ),
            ]
        ),
        html.Tr(
            [
                html.Td(
                    [
                        html.H3("Chess Board"),
                        html.Div(id="chess-board-container"),
                    ],
                    style={"width": f"{CELL_WIDTH}px", "height": f"{CELL_HEIGHT}px"},
                ),
                html.Td(
                    [
                        html.H3("Policy Heatmap"),
                        html.Div(dcc.Graph(id="policy-heatmap")),
                    ],
                    style={"width": f"{CELL_WIDTH}px", "height": f"{CELL_HEIGHT}px"},
                ),
                html.Td(
                    [
                        html.H3("Chess board with policy (Top 5)"),
                        html.Div(dcc.Graph(id="legal-moves-heatmap")),
                    ],
                    style={"width": f"{CELL_WIDTH}px", "height": f"{CELL_HEIGHT}px"},
                ),
            ]
        ),
        html.Tr(
            [
                html.Td(
                    [
                        html.H3("Chess board with policy (Top 5)"),
                        html.Div(id="chess-board-with-arrows-container"),
                    ],
                    style={"width": f"{CELL_WIDTH}px", "height": f"{CELL_HEIGHT}px"},
                ),
                html.Td(
                    [
                        html.H3("Child visit counts"),
                        html.Div(dcc.Graph(id="child-visit-counts-heatmap")),
                    ],
                    style={"width": f"{CELL_WIDTH}px", "height": f"{CELL_HEIGHT}px"},
                ),
                html.Td(
                    [
                        html.H3("Masked Policy Heatmap"),
                        html.Div(dcc.Graph(id="masked-policy-heatmap")),
                    ],
                    style={"width": f"{CELL_WIDTH}px", "height": f"{CELL_HEIGHT}px"},
                ),
            ]
        ),
    ]
)


# Callback to update the record index using buttons
@app.callback(
    Output("record-index", "value"),
    [Input("prev-button", "n_clicks"), Input("next-button", "n_clicks")],
    [Input("record-index", "value")],
)
def update_record_index(prev_clicks, next_clicks, current_index):
    if current_index is None:
        current_index = 0

    ctx = dash.callback_context
    if not ctx.triggered:
        return current_index

    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if triggered_id == "prev-button":
        current_index = max(0, current_index - 1)
    elif triggered_id == "next-button":
        current_index = min(len(dataset) - 1, current_index + 1)

    return current_index


# Callback to update the chess board, policy heatmap, legal moves heatmap, and other data
@app.callback(
    [
        Output("chess-board-container", "children"),
        Output("policy-heatmap", "figure"),
        Output("legal-moves-heatmap", "figure"),
        Output("chess-board-with-arrows-container", "children"),
        Output("child-visit-counts-heatmap", "figure"),
        Output("value", "children"),
        Output("final-value", "children"),
        Output(
            "masked-policy-heatmap", "figure"
        ),  # New output for masked policy heatmap
    ],
    [Input("record-index", "value")],
)
def update_dashboard(record_index):
    if record_index is None or record_index < 0 or record_index >= len(dataset):
        return (
            "Invalid record index",
            go.Figure(),
            go.Figure(),
            "",
            go.Figure(),
            "",
            "",
            go.Figure(),
        )

    record = dataset[record_index]

    # Start from a blank board
    board = chess.Board()
    board.clear()  # Clear all pieces from the board

    # Populate the board using channel-based logic
    board_tensor = record["board_tensor"].numpy()
    for sq in range(64):
        for channel in range(6):  # Channels 0-5 represent pieces
            piece_value = board_tensor[channel, sq // 8, sq % 8]
            if piece_value != 0:
                color = chess.WHITE if piece_value > 0 else chess.BLACK
                piece_type = channel + 1  # Channel index maps to piece type (1-6)
                board.set_piece_at(chess.Square(sq), chess.Piece(piece_type, color))
                break  # Stop checking other channels once a piece is found

    board_svg = chess.svg.board(board)
    board_svg_base64 = base64.b64encode(board_svg.encode("utf-8")).decode("utf-8")

    # Create policy heatmap using index_to_coordinates
    policy_vector = record["policy"].detach().clone().numpy()  # Fix the warning
    policy_heatmap_data = [[0 for _ in range(64)] for _ in range(64)]
    for i, value in enumerate(policy_vector):
        x, y = index_to_coordinates(i)
        policy_heatmap_data[x][y] = value

    policy_fig = go.Figure(
        data=go.Heatmap(z=policy_heatmap_data, colorscale="viridis", showscale=False)
    )
    policy_fig.update_layout(
        # title="Policy Heatmap",
        autosize=False,
        height=BOARD_SIZE,
        width=BOARD_SIZE,
        showlegend=False,
        coloraxis=dict(
            showscale=False,
        ),
        xaxis=dict(showticklabels=False, ticks=""),
        yaxis=dict(showticklabels=False, ticks=""),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Create legal moves heatmap using index_to_coordinates
    legal_mask = record["legal_mask"].detach().clone().numpy()
    legal_moves_heatmap_data = [[0 for _ in range(64)] for _ in range(64)]
    for i, value in enumerate(legal_mask):
        x, y = index_to_coordinates(i)
        legal_moves_heatmap_data[x][y] = int(value)

    legal_moves_fig = go.Figure(
        data=go.Heatmap(
            z=legal_moves_heatmap_data, colorscale="viridis", showscale=False
        )
    )
    legal_moves_fig.update_layout(
        # title="Legal Moves Heatmap",
        autosize=False,
        height=BOARD_SIZE,
        width=BOARD_SIZE,
        showlegend=False,
        coloraxis=dict(
            showscale=False,
        ),
        xaxis=dict(showticklabels=False, ticks=""),
        yaxis=dict(showticklabels=False, ticks=""),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Extract top 5 moves from policy vector
    top_moves = sorted(enumerate(policy_vector), key=lambda x: x[1], reverse=True)[:5]
    arrows = []
    for move_index, _ in top_moves:
        from_sq, to_sq = divmod(move_index, 64)
        arrows.append(
            chess.svg.Arrow(chess.Square(from_sq), chess.Square(to_sq), color="red")
        )
    board_with_arrows_svg = chess.svg.board(board, arrows=arrows)
    board_with_arrows_svg_base64 = base64.b64encode(
        board_with_arrows_svg.encode("utf-8")
    ).decode("utf-8")

    # Create child visit counts heatmap
    child_visit_counts = record["child_visit_counts"].numpy()
    child_visit_counts_heatmap_data = [[0 for _ in range(64)] for _ in range(64)]
    for i, value in enumerate(child_visit_counts):
        x, y = index_to_coordinates(i)
        child_visit_counts_heatmap_data[x][y] = value

    child_visit_counts_fig = go.Figure(
        data=go.Heatmap(
            z=child_visit_counts_heatmap_data, colorscale="viridis", showscale=False
        )
    )
    child_visit_counts_fig.update_layout(
        # title="Child Visit Counts Heatmap",
        autosize=False,
        height=BOARD_SIZE,
        width=BOARD_SIZE,
        showlegend=False,
        coloraxis=dict(
            showscale=False,
        ),
        xaxis=dict(showticklabels=False, ticks=""),
        yaxis=dict(showticklabels=False, ticks=""),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    # Create masked policy heatmap
    masked_policy_heatmap_data = [[0 for _ in range(64)] for _ in range(64)]
    for i, value in enumerate(policy_vector):
        if legal_mask[i]:  # Only include legal moves
            x, y = index_to_coordinates(i)
            masked_policy_heatmap_data[x][y] = value

    masked_policy_fig = go.Figure(
        data=go.Heatmap(
            z=masked_policy_heatmap_data, colorscale="viridis", showscale=False
        )
    )
    masked_policy_fig.update_layout(
        autosize=False,
        height=BOARD_SIZE,
        width=BOARD_SIZE,
        showlegend=False,
        coloraxis=dict(
            showscale=False,
        ),
        xaxis=dict(showticklabels=False, ticks=""),
        yaxis=dict(showticklabels=False, ticks=""),
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
    )

    value = record["value"].item()
    final_value = record["final_value"].item()

    return (
        html.Div(
            [
                html.Img(
                    src="data:image/svg+xml;base64," + board_svg_base64, width=IMG_SIZE
                )
            ]
        ),
        policy_fig,
        legal_moves_fig,
        html.Div(
            [
                html.Img(
                    src="data:image/svg+xml;base64," + board_with_arrows_svg_base64,
                    width=IMG_SIZE,
                )
            ]
        ),
        child_visit_counts_fig,
        str(value),
        str(final_value),
        masked_policy_fig,  # Add masked policy heatmap to the return values
    )


def index_to_coordinates(i):
    """Map policy index to coordinates in the 64x64 grid."""
    from_sq = i // 64  # Determine the 'from' square
    to_sq = i % 64  # Determine the 'to' square
    x_big = from_sq // 8  # Big 8x8 grid corresponds to 'from' square
    y_big = from_sq % 8  # Small 8x8 grid within each square corresponds to 'to' square
    x_small = to_sq // 8
    y_small = to_sq % 8
    x_all = x_big * 8 + x_small
    y_all = y_big * 8 + y_small
    return x_all, y_all


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
