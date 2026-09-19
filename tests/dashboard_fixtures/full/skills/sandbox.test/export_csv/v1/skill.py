def run(ctx, scope="page"):
    ctx.ctl.click(ctx.find("Export menu").box.center)
    ctx.ctl.click(ctx.find(scope.title()).box.center)
    ctx.settle()
