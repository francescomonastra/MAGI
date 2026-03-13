# Versions Guideline

## Version 1.3.sv


## Version 1.3.tv

The direction $u_v = \cos(\theta _v)$ (as $s_v = atanh(u_v)$) is modeled by a gaussian with a more complex head.
Introducing the parameter $t_v$ to smooth a bit the shape of $s_v$.

$t_v = sign(s_v)\log(1+|s_v|)$

$s_v = sign(t_v)e^{|t_v|-1}$

## Version 1.4
This version uses the $t_v$ parameter for the direction $\theta$, however, due to the difficulties in the previous version, now the model tries to fit with a mixture of gaussians:

$p (t_v|z) = \sum^K_{k=1} \pi_k \mathcal{N}(\mu _k;\sigma _k)$

where $\pi _k$ are the weights for the $K$ gaussians (suggested $K=[3;5]$)
