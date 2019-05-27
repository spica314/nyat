#[macro_use]
extern crate log;
extern crate env_logger;

use nyat_sat::sat::*;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let s = std::fs::read_to_string(args[1].as_str()).unwrap();
    let problem = SatProblem::new_from_dimacs(s.as_str());
    let mut solver = SatSolver::new(&problem);
    let assignment = solver.solve();
    eprintln!("{:?}", assignment);
    if assignment.is_some() {
        println!("SAT");
        println!("{}", assignment.unwrap().to_dimacs());
    } else {
        println!("UNSAT");
    }
}
