#[derive(Debug,Clone)]
struct SatClause(Vec<i64>);

#[derive(Debug)]
pub struct SatProblem {
    n: usize,
    clauses: Vec<SatClause>,
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
        let mut clauses = vec![];
        let mut xs = vec![];
        for ref t in iter {
            let u = t.parse::<i64>().unwrap();
            if u == 0 {
                clauses.push(SatClause(xs.clone()));
                xs.clear();
            }
            else {
                xs.push(u);
            }
        }
        assert_eq!(clauses.len(), n_clauses);
        SatProblem {
            n: n_variables,
            clauses: clauses,
        }
    }
    pub fn to_dimacs(&self) -> String {
        let mut res = String::new();
        res.push_str(&format!("p cnf {} {}\n", self.n, self.clauses.len()));
        for ref clause in &self.clauses {
            for x in &clause.0 {
                res.push_str(&format!("{} ", x));
            }
            res.push_str(&format!("0\n"));
        }
        res
    }
    fn check_assingemnt(&self, assignment: &SatAssignment) -> bool {
        for ref clause in &self.clauses {
            let mut tf = false;
            for &x in &clause.0 {
                if assignment.0[x.abs() as usize] == 0 {
                    return false;
                }
                if assignment.0[x.abs() as usize].signum() == x.signum() {
                    tf = true;
                    break;
                }
            }
            if ! tf {
                return false;
            }
        }
        true
    }
    fn add_clause(&mut self, clause: &SatClause) {
        let n2 = clause.0.iter().map(|&x| x.abs() as usize).max().unwrap();
        self.n = std::cmp::max(self.n, n2);
        self.clauses.push(clause.clone());
    }
    fn add_implies(&mut self, a: usize, b: usize) {
        let a = a as i64;
        let b = b as i64;
        self.add_clause(&SatClause(vec![-a,b]));
    }
    fn add_eq(&mut self, a: usize, b: usize) {
        self.add_implies(a,b);
        self.add_implies(b,a);
    }
    fn add_xor(&mut self, a: usize, b: usize) -> usize {
        let a = a as i64;
        let b = b as i64;
        let c = (self.n+1) as i64;
        self.add_clause(&SatClause(vec![a,b,-c]));
        self.add_clause(&SatClause(vec![-a,b,c]));
        self.add_clause(&SatClause(vec![a,-b,c]));
        self.add_clause(&SatClause(vec![-a,-b,-c]));
        c as usize
    }
    fn add_and(&mut self, a: usize, b: usize) -> usize {
        let a = a as i64;
        let b = b as i64;
        let c = (self.n+1) as i64;
        self.add_clause(&SatClause(vec![a,b,-c]));
        self.add_clause(&SatClause(vec![-a,b,-c]));
        self.add_clause(&SatClause(vec![a,-b,-c]));
        self.add_clause(&SatClause(vec![-a,-b,c]));
        c as usize
    }
    fn add_or(&mut self, a: usize, b: usize) -> usize {
        let a = a as i64;
        let b = b as i64;
        let c = (self.n+1) as i64;
        self.add_clause(&SatClause(vec![a,b,-c]));
        self.add_clause(&SatClause(vec![-a,b,c]));
        self.add_clause(&SatClause(vec![a,-b,c]));
        self.add_clause(&SatClause(vec![-a,-b,c]));
        c as usize
    }
    fn add_half_adder(&mut self, a: usize, b: usize) -> (usize,usize) {
        let c = self.add_and(a,b);
        let s = self.add_xor(a,b);
        (c,s)
    }
    fn add_hull_adder(&mut self, a: usize, b: usize, x: usize) -> (usize,usize) {
        let (c1,s1) = self.add_half_adder(a,b);
        let (c2,s)  = self.add_half_adder(s1,x);
        let c = self.add_or(c1,c2);
        (c,s)
    }
}

#[derive(Debug,PartialEq,Eq)]
pub struct SatAssignment(Vec<i64>);

pub fn solve_sat(problem: &SatProblem) -> Option<SatAssignment> {
    let mut assignment = SatAssignment(vec![0; problem.n+1]);
    loop {
        let mut updated = false;
        for ref clause in &problem.clauses {
            let mut tf = false;
            let mut unknowns = vec![];
            for &x in &clause.0 {
                if assignment.0[x.abs() as usize] == 0 {
                    unknowns.push(x);
                }
                else {
                    if assignment.0[x.abs() as usize].signum() == x.signum() {
                        tf = true;
                        break;
                    }
                }
            }
            if ! tf {
                match unknowns.len() {
                    0 => return None,
                    1 => {
                        let t = unknowns[0];
                        let i = t.abs() as usize;
                        assignment.0[i] = t.signum();
                        updated = true;
                    }
                    _ => {},
                }
            }
        }
        if ! updated {
            break;
        }
    }
    let is_sat = dfs(problem, &mut assignment, 1);
    if is_sat {
        Some(assignment)
    } else {
        None
    }
}

fn dfs(problem: &SatProblem, assignment: &mut SatAssignment, i: usize) -> bool {
    if i == problem.n+1 {
        return problem.check_assingemnt(&assignment);
    }
    if assignment.0[i] != 0 {
        return dfs(problem, assignment, i+1);
    }
    'l1: for &tmp_assign in &[1,-1] {
        assignment.0[i] = tmp_assign;
        let mut edited = vec![];
        loop {
            let mut updated = false;
            for ref clause in &problem.clauses {
                let mut tf = false;
                let mut unknowns = vec![];
                for &x in &clause.0 {
                    if assignment.0[x.abs() as usize] == 0 {
                        unknowns.push(x);
                    }
                    else {
                        if assignment.0[x.abs() as usize].signum() == x.signum() {
                            tf = true;
                            break;
                        }
                    }
                }
                if ! tf {
                    match unknowns.len() {
                        0 => {
                            for &i in &edited {
                                assignment.0[i] = 0;
                            }
                            assignment.0[i] = 0;
                            continue 'l1;
                        }
                        1 => {
                            let t = unknowns[0];
                            let i = t.abs() as usize;
                            edited.push(i);
                            assignment.0[i] = t.signum();
                            updated = true;
                        }
                        _ => {},
                    }
                }
            }
            if ! updated {
                break;
            }
        }
        let is_sat = dfs(problem, assignment, i+1);
        if is_sat {
            return true;
        }
        else {
            for &i in &edited {
                assignment.0[i] = 0;
            }
            assignment.0[i] = 0;
        }
    }
    false
}

#[test]
fn test_solve_sat_1() {
    let problem = SatProblem {
        n: 1,
        clauses: vec![
            SatClause(vec![1]),
        ],
    };
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_2() {
    let problem = SatProblem {
        n: 1,
        clauses: vec![
            SatClause(vec![-1]),
        ],
    };
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_3() {
    let problem = SatProblem {
        n: 2,
        clauses: vec![
            SatClause(vec![1,-2]),
        ],
    };
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_4() {
    let problem = SatProblem {
        n: 2,
        clauses: vec![
            SatClause(vec![-1,2]),
        ],
    };
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_5() {
    let problem = SatProblem {
        n: 2,
        clauses: vec![
            SatClause(vec![-1,2]),
        ],
    };
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_6() {
    let problem = SatProblem {
        n: 3,
        clauses: vec![
            SatClause(vec![-1,2,-3]),
        ],
    };
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_7() {
    let problem = SatProblem {
        n: 1,
        clauses: vec![
            SatClause(vec![1]),
            SatClause(vec![-1]),
        ],
    };
    let res = solve_sat(&problem);
    assert!(res.is_none());
}

#[test]
fn test_solve_sat_8() {
    let problem = SatProblem {
        n: 3,
        clauses: vec![
            SatClause(vec![1,2,-3]),
            SatClause(vec![1,-2,3]),
            SatClause(vec![-1,2,3]),
            SatClause(vec![-1,-2,-3]),
            SatClause(vec![3]),
        ],
    };
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
}

#[test]
fn test_solve_sat_9() {
    let mut problem = SatProblem {
        n: 8,
        clauses: vec![],
    };
    let mut xss = vec![vec![]; 9];
    let i_offset = 1;
    let k_offset = 5;
    for i in 0..4 {
        for k in 0..4 {
            let t = problem.add_and(i+i_offset,k+k_offset);
            xss[i+k].push(t);
        }
    }
    for i in 0..8 {
        let mut t = xss[i][0];
        for k in 1..xss[i].len() {
            let (c,s) = problem.add_half_adder(t,xss[i][k]);
            xss[i+1].push(c);
            t = s;
        }
        let t = t as i64;
        match i {
            0 => problem.add_clause(&SatClause(vec![t])),
            1 => problem.add_clause(&SatClause(vec![t])),
            2 => problem.add_clause(&SatClause(vec![t])),
            3 => problem.add_clause(&SatClause(vec![t])),
            _ => {},
        }
    }
    problem.add_clause(&SatClause(vec![-4]));
    problem.add_clause(&SatClause(vec![-8]));
    problem.add_clause(&SatClause(vec![-1,2,3,4]));
    problem.add_clause(&SatClause(vec![-5,6,7,8]));
    let res = solve_sat(&problem).unwrap();
    assert!(problem.check_assingemnt(&res));
    eprintln!("x1..=x8 = {:?}", &res.0[1..=8]);
    eprintln!("problem = {:?}", problem);
}
