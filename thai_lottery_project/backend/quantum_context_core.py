# quantum_context_core.py

def weight_superposition_with_context(superposition, context_vector, draw_metadata):
    """
    ปรับน้ำหนัก superposition โดยคำนึงถึงสัญญาณบริบท (context signals)
    เช่น info_score, cultural_score ฯลฯ
    """
    for draw_id in superposition.keys():
        ctx = draw_metadata.get(draw_id, {})
        weight = (
            1
            + context_vector.get("info_signal", 0.0) * ctx.get("info_score", 0)
            + context_vector.get("bias_signal", 0.0) * ctx.get("bias_score", 0)
            + context_vector.get("cultural_signal", 0.0) * ctx.get("cultural_score", 0)
            + context_vector.get("historical_signal", 0.0) * ctx.get("history_score", 0)
        )
        superposition[draw_id] *= weight

    # Normalize
    total = sum(superposition.values())
    return {k: v / total for k, v in superposition.items()}
