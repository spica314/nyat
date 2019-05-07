#[derive(Debug, Clone, Copy)]
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

#[derive(Debug)]
pub struct SatProblem {
    n_variables: usize,
    clauses: Clauses,
}

impl SatProblem {
    pub fn new_from_dimacs(s: &str) -> SatProblem {
        let s2 = {
            let mut res = String::new();
            for ref line in s.lines() {
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
        for ref t in iter {
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
        for ref clause in &self.clauses {
            for &literal in clause.iter() {
                let t = literal.id() + 1;
                let u = t as i64 * if literal.sign() { 1 } else { -1 };
                res.push_str(&format!("{} ", u));
            }
            res.push_str(&format!("0\n"));
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
        for i in 0..n_variables {
            assignments[i] = rng.gen::<bool>();
        }
        let mut clauses = Clauses::new();
        for _ in 0..n_clauses {
            let mut xs: Vec<Literal> = vec![];
            let dist = Uniform::from(0..n_variables);
            'l1: while xs.len() < k_sat {
                let id = dist.sample(&mut rng);
                for &x in &xs {
                    if x.id() == id {
                        continue 'l1;
                    }
                }
                let sign = if xs.len() == 0 || rng.gen::<f64>() < prob_true {
                    assignments[id]
                } else {
                    !assignments[id]
                };
                xs.push(Literal::new(id, sign));
            }
            clauses.push(Clause::new_from_vec(xs));
        }
        SatProblem {
            n_variables,
            clauses,
        }
    }
    fn check_assingemnt(&self, assignment: &SatAssignments) -> bool {
        for ref clause in &self.clauses {
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
    pub fn solve(&self) -> Option<SatAssignments> {
        let mut assignments = vec![None; self.n_variables];
        loop {
            let mut updated = false;
            for ref clause in &self.clauses {
                let mut truth_of_clause = false;
                let mut unknowns = vec![];
                for &x in &clause.0 {
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
                        0 => return None,
                        1 => {
                            let t = unknowns[0];
                            let i = t.id();
                            assignments[i] = Some(t.sign());
                            updated = true;
                        }
                        _ => {}
                    }
                }
            }
            if !updated {
                break;
            }
        }
        let is_sat = SatProblem::dfs(self, &mut assignments, 0);
        if is_sat {
            let xs: Vec<bool> = assignments.iter().map(|&x| x.unwrap()).collect();
            let res = SatAssignments::new_from_vec(xs);
            Some(res)
        } else {
            None
        }
    }

    fn dfs(problem: &SatProblem, assignments: &mut Vec<Option<bool>>, i: usize) -> bool {
        if i == problem.n_variables {
            for t in assignments.iter() {
                if t.is_none() {
                    return false;
                }
            }
            let xs: Vec<bool> = assignments.iter().map(|&x| x.unwrap()).collect();
            let assignments = SatAssignments::new_from_vec(xs);
            return problem.check_assingemnt(&assignments);
        }
        if assignments[i].is_some() {
            return SatProblem::dfs(problem, assignments, i + 1);
        }
        'l1: for &tmp_assign in &[true, false] {
            assignments[i] = Some(tmp_assign);
            let mut edited = vec![];
            loop {
                let mut updated = false;
                for ref clause in &problem.clauses {
                    let mut truth_of_clause = false;
                    let mut unknowns = vec![];
                    for &x in &clause.0 {
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
                                for &k in &edited {
                                    assignments[k] = None;
                                }
                                assignments[i] = None;
                                continue 'l1;
                            }
                            1 => {
                                let t = unknowns[0];
                                let i = t.id;
                                edited.push(i);
                                assignments[i] = Some(t.sign());
                                updated = true;
                            }
                            _ => {}
                        }
                    }
                }
                if !updated {
                    break;
                }
            }
            let is_sat = SatProblem::dfs(problem, assignments, i + 1);
            if is_sat {
                return true;
            } else {
                for &k in &edited {
                    assignments[k] = None;
                }
                assignments[i] = None;
            }
        }
        false
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct SatAssignments(Vec<bool>);

impl SatAssignments {
    fn new_from_vec(xs: Vec<bool>) -> SatAssignments {
        SatAssignments(xs)
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
    for _ in 0..1 {
        let problem = SatProblem::gen_random_sat(100, 250, 3, 0.2);
        eprintln!("problem\n{}\n", problem.to_dimacs());
        let res = problem.solve().unwrap();
        assert!(problem.check_assingemnt(&res));
    }
}
