def run(ctx, company):
    ctx.ctl.click(ctx.find("toolbar search").box.center)
    ctx.ctl.type_text(company)
