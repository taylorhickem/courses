""" your boss comes to you and tells you that we need to hire one more person in the
team, and he wants to know from you how much of your bandwidth that it will take,
in hours/week.

You ask him, when do you need them hired by? and how confident do you need me to be?
He responds --- aaah, 6 months, and 90% confident

you worked out an equation that relates the effort E (hrs/week) to the time to hire
T (weeks) and confidence P.  You defined several parameters as assumptions

E = (xi*ei)/(T-th)*log(1-P)/log(1-x)

xi = # of interviews / # CVs
x  = # of hires / # of CVs
th = time in weeks to convert one CV to a hire
ei = effort for a single interview in hrs
"""
import matplotlib.pyplot as plt
import seaborn as sbn
import pandas as pd
import numpy as np
import math

LESSON_NAME = 'Lesson 03: Numerical methods'
E_TOLERANCE = 0.0001
H_DEFAULT = 0.0001
PARAMETERS = {
    'xi': 0.5,
    'x': 0.05,  # ~ 0.5 * 0.33 * 0.33
    'th': 4,    # weeks
    'ei': 3     # hours, including offline time and evaluation, interface w HR and boss
}


def run_lesson():
    tth_months = 6
    Npts = 20
    p = 0.9
    tth_weeks = tth_months*52/12
    hpw = effort_hpw(tth_weeks, p, **PARAMETERS)
    print('Hey boss, I\'m %.1f%% confident that I\'ll need %d hours a week to hire in %d months' %
          (p*100, hpw, tth_months))

    plot_effort(p, [tth_months-3, tth_months], Npts, PARAMETERS)

    print('congratulations! \n you have completed %s' % LESSON_NAME)


# lesson part 03 Newton's method Equation 2 centered derivative (only used in derivation)
def dfdx(f, x, h=H_DEFAULT):
    centered = 0.5 * 1/h * (f(x+h) - f(x-h))
    return centered


# lesson part 03 Newton's method
def newton_rhapson(z, target, ig, h=H_DEFAULT, tol=E_TOLERANCE):
    def f(x):
        return z(x) - target

    def is_solved(g):
        loss = f(g)**2
        return loss < tol

    def newton_step(g):
        # Combination of Equation 2 centered derivative + 1st order Taylor series appx
        g_next = g - f(g) / (f(g+h)-f(g-h)) * 2 * h
        return g_next

    g = ig
    print('guess %.4f, is solved? %s' % (g, is_solved(g)))
    while not is_solved(g):
        g = newton_step(g)
        print('guess %.4f, is solved? %s' % (g, is_solved(g)))

    return g


# lesson part 03 Newton's method
def prob_hire(effort, tth, **kwargs):
    xi = kwargs['xi']
    x = kwargs['x']
    ei = kwargs['ei']
    th = kwargs['th']

    ph = 1 - (1 - x)**(effort/(xi*ei)*(tth-th))

    return ph


# lesson part 03 Newton's method
def effort_hpw_nr(tth, p, e_guess, **kwargs):
    def z(effort):
        return prob_hire(effort, tth, **kwargs)

    hpw = newton_rhapson(z, p, e_guess)

    return hpw

# lesson part 02: time to conversion using algebraic trick
def effort_hpw(tth, p, **kwargs):
    xi = kwargs['xi']
    x = kwargs['x']
    ei = kwargs['ei']
    th = kwargs['th']

    #our main equation E = (xi * ei) / (T - th) * log(1 - P) / log(1 - x)
    hpw = (xi*ei) / (tth - th) * math.log(1 - p) / math.log(1 - x)

    return hpw


def plot_effort(p, tth_range, Npts, kwargs):
    x = np.linspace(tth_range[0], tth_range[1], Npts)
    y = [effort_hpw(m * 52 / 12, p, **PARAMETERS) for m in x]
    data = pd.DataFrame({'tth_months': x, 'effort_hpw': y})

    sbn.lineplot(data, x='tth_months', y='effort_hpw')
    plt.show()


if __name__ == "__main__":
    run_lesson()