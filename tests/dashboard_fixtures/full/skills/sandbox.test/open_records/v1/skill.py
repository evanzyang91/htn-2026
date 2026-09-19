def run(ctx):
    ctx.ctl.click(ctx.find("Records nav item").box.center)
    ctx.settle()
