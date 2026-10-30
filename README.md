# Logitudal Oaxaca-Blinder decomposition

Dynamic Kitagawa-Oaxaca-Bliender decomposition of **levels** and **change in a group gap**. This is a Python port of the Stata `xtoaxaca` program written by [Jörg Hartmann](@https://gitlab.com/jhart) and [xtoaxaca](https://gitlab.com/jhart/xtoaxaca). Additionally [linearmodels](https://github.com/bashtage/linearmodels.git). The project extends original Kitagawa-Oaxaca-Blinder's approach decomposition to repeated cross-sectional or panel data.

## Supported features:
Following panel estimators are supported:
| method | linearmodels' backend |
|--------|-----------------------|
| OLS    | `panel.PooledOLS`     |
| RE     | `panel.RandomEffects` |
| FE     | `panel.PanelOLS(entity_effects=True)` |

Following change types are supported:
| Change estimator | Components | Reference |
|---------|--------|----------|
| `interventionist` | Endowments / Coefficients / Interactions | Kröger & Hartmann (2020) |
| `ssm` | Endowments / Coefficients / Interactions | Kim (2010) |
| `kim` | D1 - D5 | Kim (2010) |
| `smithwelch` | i - iv | Smith & Welch (1986) | 
| `mpjd` | Explained / Unexplained: pure / price | Makepeace et al. (1999) |
| `wellington` | w1 / w2 | Wellington (2004) |
| `interventionist_twofold` | Explained / Unexplained | Kröger & Hartmann (2020) |

## Install
via Git
```bash
pip install git+https://github.com/DmitriMt/panel_oaxaca.git
```
or via PyPi
```bash
pip install ... (to be filled)
```

## Quickstart
Example data you can get at [xtoaxaca](https://gitlab.com/jhart/xtoaxaca) (`xtoaxaca_example.dta`, `xtoaxaca_example2.dta`)
```python
import pandas as pd
from linearmodels import RandomEffects
from panel_oaxaca import panel_oaxaca

example1 = pd.read_stata('xtoaxaca_example.dta')
example1['time_clone'] = example1['time']
example1 = example1.set_index(['id', 'time'])

model = RandomEffects.from_formula('inc ~ 1 + C(group) * C(time_clone) * C(edu) + C(group) * C(time_clone) * exp', data=example1).fit()

res = panel_oaxaca(varlist=['exp', 'edu'], df=example1, model=model, groupvar='group', groupcat=('Group A', 'Group B'), timevar='time', times=[4, 2, 1], timeref=2, change=['kim'])
print(res.summary())
```

## Thank you 
to [Jörg Hartmann](@jhart) for conducting your research and for publishing Stata code!


## References
1. Kröger H, Hartmann J (2020) xtoaxaca - Extending the Kitagawa-Oaxaca-Blinder Decomposition Approach to Panel Data. doi: 10.31235/osf.io/egj79 
2. Kim, C. 2010. Decomposing the change in the wage gap between white and black men over time, 1980-2005: An extension of the Blinder-Oaxaca decomposition method. *Sociological Methods & Research* 38: 619-651. doi https://doi.org/10.1177/0049124110366235.
2. Jann, B. (2004). estout: Stata module to make regression tables. Statistical Software Components S439301, Department of Economics, Boston College. url https://ideas.repec.org/c/boc/bocode/s439301.html.
3. Makepeace, G., P. Paci, H. Joshi, and P. Dolton. 1999. How unequally has equal pay progressed since the 1970s? A study of two British cohorts. *Journal of Human Resources* 34: 534-556. doi https://doi.org/10.2307/146379.
4. Oaxaca, R. L., and M. R. Ransom. (1994). On discrimination and the decomposition of wage differentials. *Journal of Econometrics* 61: 5-21. doi https://doi.org/10.1016/0304-4076(94)90074-4.
5. Smith, J. P., and F. R. Welch. (1989). Black economic progress after Myrdal. *Journal of Economic Literature* 27: 519-564.
6. Wellington, A. J. (1993). Changes in the male/female wage gap, 1976-1985. *Journal of Human Resources* 28: 383-411. doi https://doi.org/10.2307/146209.
7. Yun, M.-S. (2005). A simple solution to the identification problem in detailed wage decompositions. *Economic Inquiry* 43: 766-772. With Erratum, *Economic Inquiry* 44: 198. doi https://doi.org/10.1093/ei/cbi053.
