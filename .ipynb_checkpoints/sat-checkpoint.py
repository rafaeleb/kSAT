# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
import random
import math


class Clause:

    def __init__(self, mask_pos, mask_neg, n):
        # Gets two numbers from [0,...,2^n - 1],
        # Interprets the "1" bits in the first as positive variables
        # And the "1" bits in the second as negative variables
        self.vars_pos = mask_pos
        self.vars_neg = mask_neg
        self.len = n
        self.tautology = False
        # Tautology
        if mask_pos & mask_neg:
            mask_pos = mask_neg = (1 << n) - 1
            self.tautology = True

    def __len__(self):
        return self.len

    def __eq__(self, other):
        pos = self.vars_pos == other.vars_pos
        neg = self.vars_neg == other.vars_neg
        n = self.len == other.len
        return pos and neg and n

    def __neg__(self):
        # Given a clause, returns the "opposite clause",
        # I.e., map x_i to !(x_i)
        return Clause(self.vars_neg, self.vars_pos, self.len)

    def __repr__(self):
        if self.tautology:
            return "True"
        s_pos = bin(self.vars_pos)[2:]
        s_neg = bin(self.vars_neg)[2:]
        st = ""
        for i in range(self.len):
            if len(s_pos) > i and s_pos[len(s_pos) - i - 1] == '1':
                st += "x" + str(i) + ", "
            if len(s_neg) > i and s_neg[len(s_neg) - i - 1] == '1':
                st += "-x" + str(i) + ", "
        return st[:-2]

    def __call__(self, *arg):
        # Gets two numbers from [0,...,2^n - 1],
        # Interprets the "1" bits in the first as positive variables
        # And the "1" bits in the second as negative variables
        if len(arg) == 2:
            assignment_pos = arg[0]
            assignment_neg = arg[1]

        if len(arg) == 1:
            assignment_pos = arg[0][0]
            assignment_neg = arg[0][1]

        pos = assignment_pos & self.vars_pos
        neg = assignment_neg & self.vars_neg
        return bool(pos or neg)

    @staticmethod
    def random(n, k=3):
        # Draw a random k-variable clause over n variables
        bits = random.sample(range(n), k)
        mask_pos = 0
        mask_neg = 0
        for bit in bits:
            if random.randint(0, 1) == 1:
                mask_pos += (1 << bit)
            else:
                mask_neg += (1 << bit)
        cl = Clause(mask_pos, mask_neg, n)
        return cl


class Formula:

    def __init__(self, clauses):
        # Gets a sequence of clauses and defines them as the formula
        assert len(set(len(clause) for clause in clauses)) <= 1
        self.clauses = [clause for clause in clauses]
        self.nof_vars = 0

        if self.clauses:
            self.nof_vars = len(self.clauses[0])

    @staticmethod
    def random(n, k, t=3):
        # Draw a random t-CNF formula over n variables with k clauses
        cl = []
        for i in range(k):
            cl += [Clause.random(n, t)]
        return Formula(cl)

    @staticmethod
    def random_assignment(n):
        # Draw a random assignment over n variables
        # We draw a random positive assignment and
        # The negative assignment is the complement
        pos = random.randint(0, (1 << n) - 1)
        neg = (1 << n) - 1 - pos
        return (pos, neg)

    @staticmethod
    def random_hashed(n):
        # Draw random log(n) bit boolean test had
        # And set i-th bit of random assignment to be
        # The inner product <had, i> over F2
        had = random.randint(0, n)

        def ip(a, b):
            return bin(a & b).count('1') % 2

        pos = neg = 0
        for i in range(n):
            if ip(i, had):
                pos += (1 << i)
            else:
                neg += (1 << i)
        return (pos, neg)

    def __repr__(self):
        st = ""
        for cl in self:
            st += str(cl) + " and "
        return st[:-5]

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __len__(self):
        return len(self.clauses)

    def __getitem__(self, i):
        assert 0 <= i < len(self)

        return self.clauses[i]

    def __next__(self):
        # Iteration method simply iterates over all clauses
        if self.iter_idx < len(self):
            res = self[self.iter_idx]
            self.iter_idx += 1
            return res
        raise StopIteration

    def __call__(self, *arg):
        # Evaluate a formula with a given assignment
        # If any clause is unsatisfied return False
        # Else, return True
        for clause in self:
            if not clause(*arg):
                return False
        return True

    def approximate_sat(self, k=1000, avg=True, hashed=False):
        # If avg, compute avg nof satisfied clauses by random assignment
        # Otherwise, compute maximum nof satisfied clauses
        max_cl = 0
        total = 0
        for i in range(k):
            sat = 0
            if hashed:
                ass = Formula.random_hashed(self.nof_vars)
            else:
                ass = Formula.random_assignment(self.nof_vars)
            for cl in self:
                if cl(ass):
                    sat += 1
            if avg:
                total += sat
            if sat > max_cl:
                max_cl = sat
        if avg:
            return total / (k * len(self))
        return max_cl / len(self)

    def approximate_count(self, k=1000, hashed=False):
        # Try k random assignments, return approximate acceptance probability
        counter = 0
        for i in range(k):
            if hashed:
                ass = Formula.random_hashed(self.nof_vars)
            else:
                ass = Formula.random_assignment(self.nof_vars)
            if self(ass):
                counter += 1
        return counter / k

    def brute_force(self, count=False):
        # Find a solution using naive brute force search
        # If count=True, count all solutions
        counter = 0
        start = 0
        end = (1 << self.nof_vars) - 1

        while start <= end:
            if self(start, end):
                if not count:
                    return (start, end)
                counter += 1
            if self(end, start):
                if not count:
                    return (end, start)
                counter += 1
            start += 1
            end -= 1

        if count:
            return counter
        return None

