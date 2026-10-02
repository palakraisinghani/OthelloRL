# Models

`oroletto.json` names the project's final trained model, **Oroletto**, which
`othello --play oroletto` loads. It will point at the final self-play
(AlphaZero-style) run once that is trained:

```json
{"path": "results/<study>/<run>/seed<k>", "board_size": 6, "about": "..."}
```

Until it exists, `othello --play oroletto` explains that and suggests the
strongest model available.
