def run(ctx, company):
    ctx.ctl.click(ctx.find("search field").box.center)
    ctx.ctl.type_text(company)
    ctx.ctl.press_key(("Enter",))
    ctx.settle()
