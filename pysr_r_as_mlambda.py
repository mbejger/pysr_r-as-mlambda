import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pysr import PySRRegressor
from pysr import jl
import sys 
import sympy
import pickle


# extra sympy mappings
extra_sympy_mappings={
    "pow_n": lambda x, n: sympy.Pow(x, n),
}  # Custom operators for sympy

# Load the dataset
def load_dataset(file_path):
    """
    Load the dataset from a given file path.
    The dataset is expected to be in a space-separated format with a header.
    """
    data = pd.read_csv(file_path, sep=' ', header=0, comment="#")

    # Split the dataset
    XM = data.iloc[:, 1].values # M column
    XL = data.iloc[:, 2].values # Lambda column 
    X = np.column_stack((XM, np.log10(XL))) # Combine M and log10(Lambda) columns

    y = data.iloc[:, 3].values # target k2 column

    return X, y

positive_constant = """
function eval_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    # See https://astroautomata.com/SymbolicRegression.jl/dev/types/#DynamicExpressions.EquationModule.Node
    is_negative_constant(node) = node.degree == 0 && node.constant && node.val::T < 0
    # (The ::T part is not required, but it just speeds it up as then Julia knows it isn't `nothing`)

    # Will walk through tree and count number of times this function is true
    num_negative_constants = count(is_negative_constant, tree)
    #  (Tree utilities are defined here: https://github.com/SymbolicML/DynamicExpressions.jl/blob/master/src/base.jl,
    #  and let you treat an expression like a Julia collection)

    if num_negative_constants > 0
        # Return 1000 times the number of negative constants as a regularization penalty
        return L(1000 * num_negative_constants)
    end

    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end
    return sum((prediction .- dataset.y) .^ 2) / dataset.n
end
"""

# Initialize PySRRegressor
model = PySRRegressor(
    loss_function=positive_constant,  # Custom loss function to penalize negative constants
    batching=True,                 # Enable mini-batch training
    batch_size=4096,               # Recommended batch size (adjust as needed)
    niterations=100,               # Number of iterations (adjust for complexity)
    binary_operators=["+", "-", "*", "/", "pow_n(x::T, n::T) where(T) = x > 0 ? convert(T, x^n) : convert(T, NaN)"],  # Operators
    extra_sympy_mappings=extra_sympy_mappings,  # Custom unary operators
    constraints={"pow_n": (-1,1)}, # Allow the model to fit the exponent "a"
    populations=150,               # Number of populations (influences exploration)
    population_size=64,            # Number of individuals in each population
    ncycles_per_iteration=100,     # Number of total mutations to run, per 10 samples of the population, per iteration
    model_selection="best",        # Keep the best-performing model
    parsimony=1e-5,                # parsimony (times complexity) added to the loss
    maxdepth=7,                    # Max depth of the equation
    progress=True,                 # Display training progress
    verbosity=1,                   # Show intermediate output
    turbo=True,                    # Use turbo mode for faster training
)

# Load training dataset
X, y = load_dataset(sys.argv[1]) 

# Fit the model to find the symbolic expression
model.fit(X, y)

# Evaluate predictions
predictions = model.predict(X)

# Load test dataset
X_test, y_test = load_dataset(sys.argv[2])
# Evaluate predictions on the test set
predictions_test = model.predict(X_test)

print("MSE:", np.mean((predictions - y)**2))
print("MSE test:", np.mean((predictions_test - y_test)**2))
print("Best equation:")
print(model.sympy())  # Prints the best equation

# Save to output directory
save_path = str(model.output_directory_) + "/" + str(model.run_id_) + "/checkpoint.pkl"
with open(save_path, "wb") as f:
    pickle.dump(model, f)


