use nyat::sat::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let s = std::fs::read_to_string(args[1].as_str()).unwrap();
    let problem = SatProblem::new_from_dimacs(s.as_str());
    let assignment = solve_sat(&problem);
    println!("{:?}", assignment);
}
