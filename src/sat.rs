const ENABLE_WATCHED_LITERALS: bool = true;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Literal {
    id: usize,
    sign: bool,
}

impl Literal {
    fn new(id: usize, sign: bool) -> Literal {
        Literal { id, sign }
    }
    fn id(&self) -> usize {
        self.id
    }
    fn sign(&self) -> bool {
        self.sign
    }
}

#[derive(Debug, Clone)]
struct Clause(Vec<Literal>);

impl Clause {
    fn new() -> Clause {
        Clause(vec![])
    }
    fn new_from_vec(xs: Vec<Literal>) -> Clause {
        Clause(xs)
    }
    fn push(&mut self, x: Literal) {
        self.0.push(x);
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn get_index(&self, id: usize) -> Option<usize> {
        for (i, literal) in self.iter().enumerate() {
            if literal.id() == id {
                return Some(i);
            }
        }
        None
    }
}

use std::convert::AsRef;
use std::iter::IntoIterator;
use std::ops::Deref;

impl IntoIterator for Clause {
    type Item = Literal;
    type IntoIter = std::vec::IntoIter<Literal>;
    fn into_iter(self) -> std::vec::IntoIter<Literal> {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Clause {
    type Item = &'a Literal;
    type IntoIter = std::slice::Iter<'a, Literal>;
    fn into_iter(self) -> std::slice::Iter<'a, Literal> {
        self.0.iter()
    }
}

impl AsRef<Clause> for Clause {
    fn as_ref(&self) -> &Clause {
        self
    }
}

impl Deref for Clause {
    type Target = [Literal];
    fn deref(&self) -> &[Literal] {
        self.0.as_slice()
    }
}

#[derive(Debug, Clone)]
struct Clauses(Vec<Clause>);

impl Clauses {
    fn new() -> Clauses {
        Clauses(vec![])
    }
    fn new_from_vec(xs: Vec<Clause>) -> Clauses {
        Clauses(xs)
    }
    fn push(&mut self, clause: Clause) {
        self.0.push(clause);
    }
    fn num(&self) -> usize {
        self.0.len()
    }
    fn iter(&self) -> std::slice::Iter<Clause> {
        self.0.iter()
    }
}

impl IntoIterator for Clauses {
    type Item = Clause;
    type IntoIter = std::vec::IntoIter<Clause>;
    fn into_iter(self) -> std::vec::IntoIter<Clause> {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Clauses {
    type Item = &'a Clause;
    type IntoIter = std::slice::Iter<'a, Clause>;
    fn into_iter(self) -> std::slice::Iter<'a, Clause> {
        self.0.iter()
    }
}

impl AsRef<Clauses> for Clauses {
    fn as_ref(&self) -> &Clauses {
        self
    }
}

impl<I: SliceIndex<[Clause]>> Index<I> for Clauses {
    type Output = <I as SliceIndex<[Clause]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<I: SliceIndex<[Clause]>> IndexMut<I> for Clauses {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[derive(Debug)]
pub struct SatProblem {
    n_variables: usize,
    clauses: Clauses,
}

impl SatProblem {
    pub fn new_from_dimacs(s: &str) -> SatProblem {
        let s2 = {
            let mut res = String::new();
            for line in s.lines() {
                let t = line.to_string();
                if t.chars().take(1).next() == Some('c') {
                    continue;
                }
                res.push_str(line);
                res.push('\n');
            }
            res
        };
        let mut iter = s2.split_whitespace();
        assert_eq!(iter.next(), Some("p"));
        assert_eq!(iter.next(), Some("cnf"));
        let n_variables = iter.next().unwrap().parse::<usize>().unwrap();
        let n_clauses = iter.next().unwrap().parse::<usize>().unwrap();
        let mut clauses = Clauses::new();
        let mut xs = vec![];
        for t in iter {
            let u = t.parse::<i64>().unwrap();
            if u == 0 {
                clauses.push(Clause(xs.clone()));
                xs.clear();
            } else if u > 0 {
                xs.push(Literal::new(u as usize - 1, true));
            } else if u < 0 {
                xs.push(Literal::new(-u as usize - 1, false));
            } else {
                unreachable!();
            }
        }
        assert_eq!(clauses.num(), n_clauses);
        SatProblem {
            n_variables,
            clauses,
        }
    }
    pub fn to_dimacs(&self) -> String {
        let mut res = String::new();
        res.push_str(&format!(
            "p cnf {} {}\n",
            self.n_variables,
            self.clauses.num()
        ));
        for clause in &self.clauses {
            for &literal in clause.iter() {
                let t = literal.id() + 1;
                let u = t as i64 * if literal.sign() { 1 } else { -1 };
                res.push_str(&format!("{} ", u));
            }
            res.push_str("0\n");
        }
        res
    }
    pub fn gen_random_sat(
        n_variables: usize,
        n_clauses: usize,
        k_sat: usize,
        prob_true: f64,
    ) -> SatProblem {
        use rand::distributions::Uniform;
        use rand::prelude::*;
        let mut assignments = vec![false; n_variables];
        let mut rng = rand::thread_rng();
        for assignment in assignments.iter_mut() {
            *assignment = rng.gen::<bool>();
        }
        let mut clauses = Clauses::new();
        for _ in 0..n_clauses {
            let mut clause = Clause::new();
            let dist = Uniform::from(0..n_variables);
            'l1: while clause.len() < k_sat {
                let id = dist.sample(&mut rng);
                for &x in &clause {
                    if x.id() == id {
                        continue 'l1;
                    }
                }
                let sign = if clause.len() == 0 || rng.gen::<f64>() < prob_true {
                    assignments[id]
                } else {
                    !assignments[id]
                };
                clause.push(Literal::new(id, sign));
            }
            clauses.push(clause);
        }
        SatProblem {
            n_variables,
            clauses,
        }
    }
    fn check_assingemnt(&self, assignment: &SatAssignments) -> bool {
        for clause in &self.clauses {
            let mut tf = false;
            for &x in &clause.0 {
                if assignment[x.id()] == x.sign() {
                    tf = true;
                    break;
                }
            }
            if !tf {
                return false;
            }
        }
        true
    }
    pub fn assign_unit_clause(&self) -> Option<Vec<Option<bool>>> {
        let mut res = vec![None; self.n_variables];
        loop {
            let mut updated = false;
            'l1: for clause in &self.clauses {
                let mut unknowns = vec![];
                for literal in clause {
                    if res[literal.id()].is_none() {
                        unknowns.push(literal);
                    } else if res[literal.id()] == Some(literal.sign()) {
                        continue 'l1;
                    }
                }
                if unknowns.len() == 0 {
                    return None;
                }
                if unknowns.len() == 1 {
                    let literal = unknowns[0];
                    res[literal.id()] = Some(literal.sign());
                    updated = true;
                }
            }
            if !updated {
                break;
            }
        }
        Some(res)
    }
    pub fn solve(&self) -> Option<SatAssignments> {
        #[derive(Debug)]
        enum AssignmentState {
            First,
            Second,
            Propageted,
        }
        let assignments = self.assign_unit_clause();
        if assignments.is_none() {
            return None;
        }
        let mut assignments = assignments.unwrap();

        let mut watch: Vec<Vec<usize>> = vec![vec![]; self.n_variables];
        let mut watched: Vec<Vec<bool>> = vec![];
        if ENABLE_WATCHED_LITERALS {
            for (clause_id, clause) in self.clauses.iter().enumerate() {
                if clause.len() >= 2 {
                    let mut xs = vec![false; clause.len()];
                    for (i, literal) in clause.iter().enumerate().take(2) {
                        xs[i] = true;
                        watch[literal.id()].push(clause_id);
                    }
                    watched.push(xs);
                } else {
                    watched.push(vec![]);
                }
            }
            assert_eq!(watched.len(), self.clauses.num());
        }

        let mut stack: Vec<(usize, AssignmentState)> = vec![];
        let n_variables = self.n_variables;
        let mut i = 0;
        while i < n_variables && assignments[i].is_some() {
            i += 1;
        }
        stack.push((i, AssignmentState::First));
        'l1: loop {
            // end(SAT)
            if i == n_variables {
                let xs: Vec<bool> = assignments.iter().map(|&x| x.unwrap()).collect();
                let res = SatAssignments::new_from_vec(xs);
                assert!(self.check_assingemnt(&res));
                return Some(res);
            }
            // try
            assert!(!stack.is_empty());
            assert_eq!(i, stack.last().unwrap().0);
            match stack.last().unwrap().1 {
                AssignmentState::First => {
                    assignments[i] = Some(false);
                }
                AssignmentState::Second => {
                    assignments[i] = Some(!assignments[i].unwrap());
                }
                AssignmentState::Propageted => {
                    panic!();
                }
            }

            // unit propagation
            use std::collections::VecDeque;
            let mut unit_propagation_stack = VecDeque::new();
            unit_propagation_stack.push_back(i);
            use std::collections::BTreeSet;
            let mut visited = BTreeSet::new();
            while let Some(id) = unit_propagation_stack.pop_back() {
                if visited.contains(&id) {
                    continue;
                }
                visited.insert(id);
                if ENABLE_WATCHED_LITERALS {
                    let visit_clause_ids: Vec<usize> = watch[id].clone();
                    for &clause_id in &visit_clause_ids {
                        let prev_i_literal = self.clauses[clause_id].get_index(id);
                        assert!(prev_i_literal.is_some());
                        let prev_i_literal = prev_i_literal.unwrap();
                        if !watched[clause_id][prev_i_literal] {
                            continue;
                        }
                        if self.clauses[clause_id][prev_i_literal].sign()
                            == assignments[id].unwrap()
                        {
                            continue;
                        }
                        let mut next_i_literal = None;
                        let mut next_literal_id = None;
                        for (i_literal, literal) in self.clauses[clause_id].iter().enumerate() {
                            if literal.id() != id
                                && assignments[literal.id()] != Some(!literal.sign())
                                && !watched[clause_id][i_literal]
                            {
                                next_i_literal = Some(i_literal);
                                next_literal_id = Some(literal.id());
                            }
                        }
                        if let Some(next_i_literal) = next_i_literal {
                            let next_literal_id = next_literal_id.unwrap();
                            assert!(id != next_literal_id);
                            assert!(watched[clause_id][prev_i_literal]);
                            watched[clause_id][prev_i_literal] = false;
                            assert!(!watched[clause_id][next_i_literal]);
                            watched[clause_id][next_i_literal] = true;
                            watch[id] = watch[id]
                                .iter()
                                .filter(|&&x| x != clause_id)
                                .cloned()
                                .collect();
                            watch[next_literal_id].push(clause_id);
                        } else {
                            for (i_literal, literal) in self.clauses[clause_id].iter().enumerate() {
                                if watched[clause_id][i_literal] && literal.id() != id {
                                    let id2 = literal.id();
                                    if assignments[id2].is_none() {
                                        assignments[id2] = Some(literal.sign());
                                        stack.push((id2, AssignmentState::Propageted));
                                        unit_propagation_stack.push_back(id2);
                                    } else if assignments[id2].unwrap() != literal.sign() {
                                        // conflict
                                        while let Some((k, state)) = stack.pop() {
                                            match state {
                                                AssignmentState::First => {
                                                    i = k;
                                                    stack.push((k, AssignmentState::Second));
                                                    continue 'l1;
                                                }
                                                AssignmentState::Second => {
                                                    assignments[k] = None;
                                                }
                                                AssignmentState::Propageted => {
                                                    assignments[k] = None;
                                                }
                                            }
                                        }
                                        // UNSAT
                                        return None;
                                    }
                                }
                            }
                        }
                    }
                }

                let watch_id_ids: Vec<usize> = if ENABLE_WATCHED_LITERALS {
                    watch[id].clone()
                } else {
                    (0..self.clauses.num()).collect()
                };
                for &clause_id in &watch_id_ids {
                    let clause = &self.clauses[clause_id];
                    assert!(!ENABLE_WATCHED_LITERALS || clause.iter().any(|x| x.id() == id));
                    let mut truth_of_clause = false;
                    let mut unknowns = vec![];
                    for &x in clause {
                        if let Some(assign) = assignments[x.id()] {
                            if assign == x.sign() {
                                truth_of_clause = true;
                                break;
                            }
                        } else {
                            unknowns.push(x);
                        }
                    }
                    if !truth_of_clause {
                        match unknowns.len() {
                            0 => {
                                // conflict
                                while let Some((k, state)) = stack.pop() {
                                    match state {
                                        AssignmentState::First => {
                                            i = k;
                                            stack.push((k, AssignmentState::Second));
                                            continue 'l1;
                                        }
                                        AssignmentState::Second => {
                                            assignments[k] = None;
                                        }
                                        AssignmentState::Propageted => {
                                            assignments[k] = None;
                                        }
                                    }
                                }
                                // UNSAT
                                return None;
                            }
                            1 => {
                                if ENABLE_WATCHED_LITERALS && clause.len() >= 2 {
                                    panic!();
                                }
                                let t = unknowns[0];
                                let id2 = t.id();
                                assignments[id2] = Some(t.sign());
                                stack.push((id2, AssignmentState::Propageted));
                                unit_propagation_stack.push_back(id2);
                            }
                            _ => {}
                        }
                    }
                }
            }
            while i < n_variables && assignments[i].is_some() {
                i += 1;
            }
            if i < n_variables {
                stack.push((i, AssignmentState::First));
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct SatAssignments(Vec<bool>);

impl SatAssignments {
    fn new_from_vec(xs: Vec<bool>) -> SatAssignments {
        SatAssignments(xs)
    }
    pub fn to_dimacs(&self) -> String {
        let mut res = String::new();
        for i in 0..self.0.len() {
            let x = self.0[i];
            if !x {
                res.push_str(&format!("-{} ", i + 1));
            } else {
                res.push_str(&format!("{} ", i + 1));
            }
        }
        res.push_str("0");
        res
    }
}

use std::ops::Index;
use std::ops::IndexMut;
use std::slice::SliceIndex;

impl<I: SliceIndex<[bool]>> Index<I> for SatAssignments {
    type Output = <I as SliceIndex<[bool]>>::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.0[index]
    }
}

impl<I: SliceIndex<[bool]>> IndexMut<I> for SatAssignments {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.0[index]
    }
}

#[test]
fn test_solve_sat_1() {
    let problem = SatProblem {
        n_variables: 1,
        clauses: Clauses::new_from_vec(vec![Clause::new_from_vec(vec![Literal::new(0, true)])]),
    };
    let res = problem.solve().unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_2() {
    let problem = SatProblem {
        n_variables: 1,
        clauses: Clauses::new_from_vec(vec![Clause::new_from_vec(vec![Literal::new(0, false)])]),
    };
    let res = problem.solve().unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_3() {
    let problem = SatProblem {
        n_variables: 2,
        clauses: Clauses::new_from_vec(vec![Clause::new_from_vec(vec![
            Literal::new(0, true),
            Literal::new(1, false),
        ])]),
    };
    let res = problem.solve().unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_4() {
    let problem = SatProblem {
        n_variables: 2,
        clauses: Clauses::new_from_vec(vec![Clause::new_from_vec(vec![
            Literal::new(0, false),
            Literal::new(1, true),
        ])]),
    };
    let res = problem.solve().unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_5() {
    let problem = SatProblem {
        n_variables: 2,
        clauses: Clauses::new_from_vec(vec![Clause::new_from_vec(vec![
            Literal::new(0, false),
            Literal::new(1, true),
        ])]),
    };
    let res = problem.solve().unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_6() {
    let problem = SatProblem {
        n_variables: 3,
        clauses: Clauses::new_from_vec(vec![Clause::new_from_vec(vec![
            Literal::new(0, false),
            Literal::new(1, true),
            Literal::new(2, false),
        ])]),
    };
    let res = problem.solve().unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_7() {
    let problem = SatProblem {
        n_variables: 1,
        clauses: Clauses::new_from_vec(vec![
            Clause::new_from_vec(vec![Literal::new(0, true)]),
            Clause::new_from_vec(vec![Literal::new(0, false)]),
        ]),
    };
    let res = problem.solve();
    assert!(res.is_none());
}

#[test]
fn test_solve_sat_8() {
    let problem = SatProblem {
        n_variables: 3,
        clauses: Clauses::new_from_vec(vec![
            Clause::new_from_vec(vec![
                Literal::new(0, true),
                Literal::new(1, true),
                Literal::new(2, false),
            ]),
            Clause::new_from_vec(vec![
                Literal::new(0, true),
                Literal::new(1, false),
                Literal::new(2, true),
            ]),
            Clause::new_from_vec(vec![
                Literal::new(0, false),
                Literal::new(1, true),
                Literal::new(2, true),
            ]),
            Clause::new_from_vec(vec![
                Literal::new(0, false),
                Literal::new(1, false),
                Literal::new(2, false),
            ]),
            Clause::new_from_vec(vec![Literal::new(2, true)]),
        ]),
    };
    let res = problem.solve().unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_9() {
    for _ in 0..1000 {
        let problem = SatProblem::gen_random_sat(10000, 10000, 4, 0.2);
        // eprintln!("problem\n{}\n", problem.to_dimacs());
        let res = problem.solve().unwrap();
        assert!(problem.check_assingemnt(&res));
    }
}
