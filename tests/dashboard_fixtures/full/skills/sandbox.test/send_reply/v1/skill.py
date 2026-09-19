def run(ctx, body):
    ctx.ctl.click(ctx.find("Reply button").box.center)
    ctx.ctl.type_text(body)
    ctx.ctl.press_key(("Meta", "Enter"))
    ctx.settle()
