def detect_mispricing(model_probs, market_probs, odds, threshold=0.03):
    bets = []
    labels = ["H", "D", "A"]

    for i in range(3):
        edge = model_probs[i] - market_probs[i]
        ev = model_probs[i] * odds[i] - 1

        if edge > threshold:
            bets.append({
                "side": labels[i],
                "edge": edge,
                "ev": ev,
                "odds": odds[i]
            })

    return bets