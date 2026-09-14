# Architecture notes

Short answers to the questions a reviewer asks first.

## Why is the statistics engine Python, when the UI is TypeScript?

Because it has to run in both places and be the same code in both. scipy is the
only library that gives exact noncentral t and F distributions, exact Poisson
intervals, `permutation_test`, BCa bootstrap and a reliable stiff ODE solver in
one permissively licensed package — and Pyodide ships it. Reimplementing that in
TypeScript would have meant a second implementation to keep correct, and the
first thing to go wrong would have been a number that differed between the
browser and the server, which is precisely the failure mode this platform exists
to make impossible.

## Why not just compute on the server?

A forty-minute period, a shared lab PC, and a metered line. Also: data that never
leaves the device is data the platform never processed, which under the DPDP Act
is a much stronger position than processing it carefully.

## Why is `justification` required on every inferential tool?

It is the input to the specification-search detector, it is what a trainer reads
in the review queue, and it forces the student to state a reason about the design
before seeing an answer. It is also the cheapest possible intervention: one
sentence, and it changes the shape of the question being asked.

## Why does the ledger key on a data fingerprint rather than a session?

So that re-uploading the same numbers under a new name does not reset the count.
The point of the ledger is that the student watches it climb.

## Why is the stability check automatic rather than a menu item?

Because a model's stated confidence is a fluency signal, not a stability
measurement, and the only way to know whether a conclusion holds is to perturb
the data and re-run it. An optional check is one nobody runs on the result they
like.

## Why is Ollivier-Ricci curvature hand-implemented?

`GraphRicciCurvature` v0.6.1 pins `scipy<=1.13.1` — which collides with the
version the rest of the engine needs — and pulls in `networkit`, a C++ build that
is the real installation cost. The curvature of one edge is an optimal-transport
problem over a neighbourhood of at most k+1 points; `scipy.optimize.linprog`
solves it exactly in well under a millisecond. Removing the dependency removed
the pin, the build, and a Pyodide blocker at once.

## Why does `bridge_analysis` run both weightings every time?

Because which one you used changes the answer, and a curvature reported without
its weighting, its k, its alpha and its solver is not a result. Running both and
printing the comparison is cheaper than trusting anyone to remember.

## Where would you look first if something were wrong?

`engine/tests/test_honesty.py`. If those pass, the platform's central claim
holds. Everything else is features.
