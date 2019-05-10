use nyat::sat::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let s = std::fs::read_to_string(args[1].as_str()).unwrap();
    let problem = SatProblem::new_from_dimacs(s.as_str());
    let assignment = problem.solve();
    eprintln!("{:?}", assignment);
    if assignment.is_some() {
        println!("SAT");
        println!("{}", assignment.unwrap().to_dimacs());
    } else {
        println!("UNSAT");
    }
}
