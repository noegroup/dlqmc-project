import uncertainties


def short_fmt(x):
    return f'{x:S}' if isinstance(x, uncertainties.core.AffineScalarFunc) else x
