from pysr import PySRRegressor
import sympy
import sys 

# extra sympy mappings
extra_sympy_mappings={
    "pow_n": lambda x, n: sympy.Pow(x, n),
}  # Custom operators for sympy

run_directory = "./outputs/" + sys.argv[1]    # model directory

model = PySRRegressor.from_file(
    run_directory=run_directory, 
    extra_sympy_mappings=extra_sympy_mappings
)   # read model from directory

print("#Equation:", model.sympy())   # print best equation
 
