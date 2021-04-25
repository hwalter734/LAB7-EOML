def cost(Y, X, T):
    return(((X @ T.T - Y) ** 2) * ((Y != 0) * 1)).sum()


def gradient(Y, X, T):
    R = (Y != 0) * 1
    hip_error = (X @ T.T - Y) * R

    return (
        hip_error @ T,
        hip_error.T @ X,
    )


def adam(
        Y,
        Xo,
        To,
        fun,
        jac,
        alpha=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=0.0000001,
        max_iter=1000
):
    xm = cp.zeros(Xo.shape)
    tm = cp.zeros(To.shape)

    xv = cp.zeros(Xo.shape)
    tv = cp.zeros(To.shape)

    X, T, t = Xo, To, 0

    while t < max_iter:
        t += 1

        xg, tg = jac(Y, X, T)

        print(f'{t} \t loss={fun(Y, X, T).item():,.2f}')

        xm = beta1 * xm + (1 - beta1) * xg
        tm = beta1 * tm + (1 - beta1) * tg

        xv = beta2 * xv + (1 - beta2) * xg * xg
        tv = beta2 * tv + (1 - beta2) * tg * tg

        xmh = xm / (1 - beta1 ** t)
        tmh = tm / (1 - beta1 ** t)

        xvh = xv / (1 - beta2 ** t)
        tvh = tv / (1 - beta2 ** t)

        X -= alpha * xmh / (cp.sqrt(xvh) + epsilon)
        T -= alpha * tmh / (cp.sqrt(tvh) + epsilon)

    return X, T
