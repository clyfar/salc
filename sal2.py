#!/usr/bin/env python

# Author: Geoffrey Golliher
#
# Salary modeling based on multivaraiate inputs.

import contextlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optparse
import random
import sys
import StringIO

@contextlib.contextmanager
def stdoutIO(stdout=None):
  old = sys.stdout
  if stdout is None:
    stdout = StringIO.StringIO()
  sys.stdout = stdout
  yield stdout
  sys.stdout = old

class DataContainerError(Exception):
  """Custom exception class for DataContainer."""
  def __init__(self, value):
    self.value = value

  def __str__(self):
    return repr(self.value)

class DataContainer(object):
  """This is just kind of a fun experiment setting up an iterator
     type object. The idea is to have something that works kind of
     like collections.namedtuple. Instead of creating this object once
     and letting the object create new selfs, one has to create new
     DataContainers and call each time. Not sure if there is any downside
     to this considering the same process is essentially happening
     under the covers in namedtuple. This way is less memory efficient but
     the loss is negligible (for small invocations). This is compatible with
     Python 2.4 and higher.

     Can be used (more readable). Where interp is a list of
     DataContainer objects:
       ["12/30/%s,%.2f,%.2f,%.2f,%.2f" %
         (i.year,i.sal,i.ret,i.taxable,i.nontax) for i in interp]

      or using the __iter__ (more compact)

       ["12/30/%s,%.2f,%.2f,%.2f,%.2f" % (v,w,x,y,z) for v,w,x,y,z in interp]
  """
  def __init__(self, *args):
    # Initialize our current index to 0.
    self.current = 0
    # Making the initial named keys. Args can be tuple or list.
    setattr(self, 'nkeys', list(*args))
    # Making all keys local attributes set to None.
    for k in list(*args):
      setattr(self, k, None)

  def __call__(cls, *args):
    # Assigning local attributes found in self.nkeys to the
    # corresponding value. Returns the populated DataContainer object.
    for i in xrange(0, len(list(*args))):
      setattr(cls, cls.nkeys[i], list(*args)[i])
    return cls

  def __iter__(self):
    # Just return self because we have a next() method.
    return self

  def next(self):
    if self.current > len(self.nkeys):
      # We're done iterating ... theoretically, we shouldn't reach this line.
      raise StopIteration
    else:
      # precrement.
      self.current += 1
      try:
        return getattr(self, self.nkeys[self.current - 1])
      # We're done iterating.
      except IndexError:
        raise StopIteration
      # We actually have a problem.
      except AttributeError as e:
        raise DataContainerError(e)
      # We have an unexpected problem QQ.
      except Exception as e:
        raise DataContainerError(e)
    # If we got here, we have a problem but I'm not sure what it will be.
    raise DataContainerError('ValueError')

class SalaryCalculator(object):

  year = None
  salary = None

  def __init__(self, year, salary):
    self.year = year
    self.salary = float(salary)

  def adjustSalary(self, type, adjustment):
    return eval('%.2f %s %.2f' % (self.salary, type, float(adjustment)))

  def adjustNum(self, base, type, adjustment):
    return eval('%.2f %s %.2f' % (float(base), type, float(adjustment)))

  def mean(self, l):
    ar = np.array(l)
    return ar.mean()

  def randomizeVariate(self, mean, deviation):
    return random.normalvariate(float(mean), float(deviation))

  def getNormalizedRandom(self, mean, deviation, iters=5):
    basex = []
    basey = []
    for i in xrange(iters):
      basex.extend([self.randomizeVariate(mean, deviation)])
      basey.extend([mean])
    bx = np.array(basex)
    by = np.array(basey)
    A = np.vstack([bx, np.ones(len(bx))]).T
    # For later. We want to fit our randomized estimates to the linear
    # model for whatever variate we're tracking.
    m = np.linalg.lstsq(A, by)[0]
    return np.mean(bx)

def main():
  # Temp var for transient new salary.
  ns = None
  # Temp var for transient new 401k balance.
  pp = None
  # Will be populated with DataContainers (nt ...
  # Formerly namedtuples, hence the name).
  interp = []

  parser = optparse.OptionParser(
      """Usage: %prog [options]

      Example:
      %prog -c --raise-interval 4 --raise-percent 1.04 --salary-growth-percent 1.02 \\
      --salary-growth-interval 2 --max-step 0.08 --beginning-401-balance 17000 \\
      --employer-match 5000 --starting-salary 129000 --market-growth 1.08 --salary-percent 0.04
      """,
       version="%prog .01")

  parser.add_option("-c", "--csv", action="store_true", dest="csv_on",
                    help="Specify csv output.", default=False)

  parser.add_option("-s", "--step", dest="step",
                    help="Define annual 401k step percentage.", default=0.0)

  parser.add_option("-m", "--max-step", dest="maxstep",
                    help="Define maximum annual 401k step percentage.", default=0.12)

  parser.add_option("-i", "--raise-interval", dest="sraise",
                    help="Define years between annual raise (not inflation increase).",
            default=None)

  parser.add_option("-r", "--raise-percent", dest="praise",
                    help="Define annual raise percent (not inflation increase --salary-growth-percent).",
            default=1.08)

  parser.add_option("-g", "--salary-growth-percent", dest="sg",
                    help="Define salary growth percent (inflation raises).", default=None)

  parser.add_option("-w", "--inflation-rate", dest="inflation_rate",
                    help="Define inflation rate.", default=None)

  parser.add_option("-k", "--salary-growth-interval", dest="sgi",
                    help="Define salary growth interval (inflation raises).", default=None)

  parser.add_option("-b", "--beginning-401-balance", dest="bb",
                    help="Define beginning 401k balance.", default=0)

  parser.add_option("-e", "--employer-match", dest="em",
                    help="Define employer match.", default=0)

  parser.add_option("-y", "--start-year", dest="sy",
                    help="Define starting year.", default=2016)

  parser.add_option("-n", "--years", dest="y",
                    help="Define number of years.", default=25)

  parser.add_option("-o", "--starting-salary", dest="si",
                    help="Define number of years.", default=50000)

  parser.add_option("-z", "--market-growth", dest="bg",
                    help="Define 401k market growth.", default=1.04)

  parser.add_option("-p", "--salary-percent", dest="ps",
                    help="Define percent of salary contributed to 401k.",
            default=0.05)

  parser.add_option("-t", "--stock-options", dest="tso",
                    help="Define the number of stock options received.",
            default=None)

  parser.add_option("-q", "--stock-growth", dest="stock_growth",
                    help="Define the stock option growth per year.",
            default=None)

  parser.add_option("-x", "--local", dest="local", action="store_true",
                    help="Output a graph.",
            default=False)

  parser.add_option("-a", "--recesion-interval", dest="recession_int", action="store_true",
                    help="Years between recessions.",
            default=8)

  (options, args) = parser.parse_args()

  sc = SalaryCalculator(options.y, float(options.si))
  # This is just a lazy copy of options.ps (--salary-percent).
  ps = float(options.ps)
  # Another lazy copy of an options var.
  max_step = float(options.maxstep)

  if not options.csv_on:
    print "Begining balance: %s.\n\n" % options.bb

  # Set up raise counters (rc = raise counter, sgc = salary growth counter).
  rc = 0
  sgc = 0
  # Temp var for holding the transient raise value.
  sal_raise = 0
  os = 0
  pop = 0
  linfl = 0.0
  recession_adjusted_pp = 0
  recession_counter = 0
  # Initial inflation rate.
  infl = 0.0
  # These two lists are for calculating the mean for inflation and market
  # growth. Use these to make sure we're not swinging wildly off track.
  # The means should be within a few points of the base passed in by flag.
  infl_list=[]
  mg_list=[]
  # Testing weighting ... while this variable exists, options.sgi
  # is broken. However, typically, inflation raises are given out between
  # 1-4 years.
  x = [1,2,2,3,3,3,3,3,4]
  random.shuffle(x)
  sgi = random.sample(x, 1)[0]
  # Each year gets a SalaryCalculator and attributes describing what sort of
  # transforms are requested? Seems a bit heavy handed but makes things very
  # flexible ... maybe there is a more efficient way.
  for i in xrange(0, int(options.y)+1):
    # Making the market growth rate a normalized variable with stochastic properties.
    # The normalvariate parameters are mean(growth_rate), standard deviation. The
    # standard deviation is also randomized to allow less predictability with respect
    # to market variations. The standard deviation for market growth is just a best
    # guess based on various papers.
    if options.bg:
      bg = sc.getNormalizedRandom(options.bg, sc.randomizeVariate(-0.05, 0.01))
      mg_list.append((bg - 1))
    # Making the inflation rate a normalized variable with stochastic properties.
    # The normalvariate parameters are mean(inflation_rate), standard deviation. The
    # standard deviation is also randomized to allow less predictability with respect
    # to inflation variations. The standard deviation for inflation is just a best
    # guess based on various papers.
    if options.inflation_rate:
      infl = sc.getNormalizedRandom(
          options.inflation_rate, sc.randomizeVariate(-0.05, 0.01))
      infl_list.append(infl)
    if not linfl:
      linfl = infl
    if not ns:
      ns = float(options.si)
      if options.inflation_rate:
        os = ns
    else:
      # If there is an option for a raise interval and the raise counter (rc)
      # is the current year, calculate the new salary plus inflation raise. OR
      # if a merit raise percent is defined and no interval is defined, calculate
      # the new salary plus inflation raise. If there is no interval defined, the
      # new salary will be calculated each year.
      if (options.sraise and rc == int(options.sraise)) or (options.praise and not options.sraise):
        if options.sg:
          sal_raise = sc.getNormalizedRandom(
            sc.adjustNum(
              options.praise, '+', (sc.adjustNum(options.sg, '/', 100))),
            0.0001)
          ns *= sal_raise
        else:
          sal_raise = sc.getNormalizedRandom(options.praise, 0.001)
          ns *= sal_raise
        # Reset all counters for merit raises (--raise-interval, --raise-percent).
        rc =  0
        sgc = 1
    # If no raise options were defined, try to calculate the new salary based
    # on the inflation rate salary increases (--salary-growth-percent,
    # --salary-growth-interval).
    # If the salary growth option is set:
    if options.sg:
    # If the salary growth interval is set, calculate the new salary based on
    # inflation (--salary-growth-percent).
      if options.sgi and sgc == sgi:
        sal_raise = sc.getNormalizedRandom(options.sg, 0.001)
        ns *= sal_raise
        # Reset salary growth counter.
        sgc = 1
        sgi = random.sample(x, 1)[0]
    # If the salary growth interval is not set, just calculate the new salary
    # based on inflation (--salary-growth-percent) each year.
    elif not options.sgi and options.sg:
      sal_raise = sc.getNormalizedRandom(options.sg, 0.001)
      ns *= sal_raise
    if options.inflation_rate:
      os = ns
      os -= ns * infl
    if not options.csv_on:
      print "------- For year %s ------ " % i
      print "Salary = %.2f with a %.2f percent raise.\n" % (ns, sal_raise)
      sal_raise = 0.0
      ls = ns
    if not pp:
    # The new balance has not been calculated yet. Take the initial salary and calculate
    # the next year's balance based on the defined market growth, employer matching and
    # the beginning 401k balance.
      pp = float(options.bb) + ((float(options.si) * ps) + float(options.em))
      # Inflation based 401k balance variable.
      if options.inflation_rate:
        pop = float(options.bb) + ((float(options.si) * ps) + float(options.em))
      # If there is a max_step defined, the percent of salary contributed to 401k will be
      # incremented so long as the new percent is less than the max_step (--max-step).
      if ps < max_step:
        ps += float(options.step)
    else:
      # The previous 401k balance has been calculated before. So now calculate the new
      # moving balance.
      pp = (pp + (ns * float(ps) + float(options.em))) * bg
      # Calculate the new balance adjusted for inflation. The balance is calculated using
      # inflation adjusted salary numbers ... does the overall balance have to be adjusted
      # again?
      if options.inflation_rate:
        pop = (pop + (ns * float(ps) + float(options.em))) * (bg - infl)
        # Factor in recessions for both retirement and retirement future dollars.
      if options.recession_int and recession_counter == options.recession_int and options.inflation_rate:
        pp = (pp + (ns * float(ps) + float(options.em))) * (bg - infl - 0.08)
        pop = (pop + (ns * float(ps) + float(options.em))) * (bg - infl - 0.08)
        recession_counter = 0


    # If there is a max_step defined, the percent of salary contributed to 401k will be
    # incremented so long as the new percent is less than the max_step (--max-step).
    if ps < max_step:
      ps += float(options.step)
    if not options.csv_on:
      print "401k bal = %.2f with %.2f percent growth." % (pp, bg)
      print "-------- End year %s ------\n\n" % i
    # Calculate taxable salary (salary minus 401k contributions).
    ts = int(ns) - (float(options.si) * ps + float(options.em))
    # Calculate non-taxable salary (the difference between the new salary and the taxable salary).
    txs = int(ns) - ts
    if sal_raise > 1:
      sal_raise = (sal_raise - 1) * 100
    if bg > 1:
      bg = (bg - 1) * 100
    if options.inflation_rate:
      nt = DataContainer(('year', 'sal', 'sg', 'salfd', 'infl', 'ret', 'mg', 'retfd'))
      interp.append(nt((int(options.sy)+i, ns, sal_raise, os, infl, pp, bg, pop)))
      sal_raise = 0.0
    else:
      nt = DataContainer(('year', 'sal', 'ret', 'taxable', 'nontax'))
      interp.append(nt((int(options.sy)+i, ns, pp, ts, txs)))
    # Increment all raise counters.
    rc += 1
    sgc += 1
    recession_counter += 1
  if not options.csv_on:
    print "Ending balance for year %s: %.2f" % (int(options.sy) + i, pp)

  # Interactive graph matplotlib
  pl = {}
  pl['sal'] = []
  pl['salf'] = []
  pl['ret'] = []
  pl['retfd'] = []
  if options.local and options.inflation_rate:
    plottpl = [(a,b,c,d,e,f,g,h) for a,b,c,d,e,f,g,h in interp]
    for i in xrange(len(plottpl)):
      pl['sal'].append(plottpl[i][1])
      pl['salf'].append(plottpl[i][3])
      pl['ret'].append(plottpl[i][5])
      pl['retfd'].append(plottpl[i][7])
    plt.plot(pl['sal'], label="Salary")
    plt.plot(pl['salf'], label="Salary Future Dollars")
    plt.plot(pl['ret'], label="401k Balance")
    plt.plot(pl['retfd'], label="401k Balance Future Dollars")
    plt.legend(loc=2)
    plt.ylabel("Dollars")
    plt.xlabel("Year")
    labels = [i for i in xrange(options.sy, options.sy+options.y, 1)]
    plt.xticks(np.arange(25),labels,rotation=30)
    plt.show()
    exit()
  if options.csv_on:
    if options.inflation_rate:
      print "Year,Salary,Salary Growth,Salary Future Dollars,Inflation Rate,401k Balance,Market Growth,401k Balance Future Dollars"
      print '\n'.join(["12/30/%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f" % (v,w,x,y,(z * 100),a,b,c) for v,w,x,y,z,a,b,c in interp])
    else:
      print "Year,Salary,401k Balance,Taxable Salary,Non-Taxable 401k"
      print '\n'.join(["12/30/%s,%.2f,%.2f,%.2f,%.2f" % (v,w,x,y,z) for v,w,x,y,z in interp])


if __name__ == "__main__":
  main()
