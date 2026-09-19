def verify(ctx, result):
    return ctx.index.find_text("Sent") != []
