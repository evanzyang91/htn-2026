def run(ctx, row=1):
    rows = ctx.index.by_kind("row")
    ctx.ctl.click(rows[row - 1].box.center)
    ctx.settle()
