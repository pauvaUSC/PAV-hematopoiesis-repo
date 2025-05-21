import numpy as np
import random

# Default parameters for the hematopoiesis model
params = {
    # Division thresholds: Maximum number of divisions before a cell becomes inactive
    "LT_HSC_DIVISION_THRESHOLD": 10,  # Long-term HSC division threshold
    "ST_HSC_DIVISION_THRESHOLD": 3,  # Short-term HSC division threshold

    # Apoptosis probabilities for non-quiescent cells
    "PAA": 0.0025,  # Probability of apoptosis for LT-HSCs
    "PAB": 0.0025,  # Probability of apoptosis for ST-HSCs

    # Apoptosis probabilities for quiescent cells
    "PAQA": 0.0,  # Probability of apoptosis for quiescent LT-HSCs
    "PAQB": 0.0,  # Probability of apoptosis for quiescent ST-HSCs

    # Probabilities of entering quiescence
    "PQA": 0.05,  # Probability of LT-HSCs entering quiescence
    "PQB": 0.15,  # Probability of ST-HSCs entering quiescence

    # Probabilities of exiting quiescence
    "P_EXIT_QUIESCENCE_A": 0.05,  # LT-HSCs exiting quiescence
    "P_EXIT_QUIESCENCE_B": 0.15,  # ST-HSCs exiting quiescence

    # Division fate probabilities for LT-HSCs
    "P1A": 0.60,    # Symmetric Self-Renewal: LT-HSC -> 2 LT-HSCs
    "P2A": 0.3090,    # Asymmetric Division: LT-HSC -> LT-HSC + ST-HSC
    "P3A": 0.0147,    # Symmetric Differentiation: LT-HSC -> 1 ST-HSC + death
    "P4A": 0.0763,    # Full Differentiation: LT-HSC -> 2 ST-HSCs + death

    # Division fate probabilities for ST-HSCs
    "P1B": 0.50,    # Symmetric Self-Renewal: ST-HSC -> 2 ST-HSCs
    "P2B": 0.0117,    # Asymmetric Division: ST-HSC -> ST-HSC + MPP
    "P3B": 0.216,    # Symmetric Differentiation: ST-HSC -> 1 MPP + death
    "P4B": 0.2723,    # Full Differentiation: ST-HSC -> 2 MPPs + death

    # Division timing parameters for LT-HSCs
    "LT_HSC_TIME_MEAN": 40,   # Mean time to divide (days)
    "LT_HSC_TIME_STD": 19.6,  # Standard deviation of division time (days)

    # Division timing parameters for ST-HSCs
    "ST_HSC_TIME_MEAN": 12,   # Mean time to divide (days)
    "ST_HSC_TIME_STD": 3.8,   # Standard deviation of division time (days)

    # Labeling parameters
    "LABEL_THRESHOLD": 0.06,  # Threshold for determining if a cell is labeled
    "LABELING_PERIOD": 1,     # Time period during which labeling occurs (days)
    "LABELING_PROB": 0.01,   # Probability of labeling a cell during the labeling period
    "LABELING_FRACTION": 0.01,  # Fraction of LT-HSCs to label initially
    "LABELING_DELAY": 30      # Delay (in steps) before labeling starts
}

# Global counter for unique cell IDs
cell_id_counter = 0

class Cell:
    def __init__(self, cell_type, state, age, unique_id, params, lt_rng, st_rng, mpp_rng, parent_id=None, division_count=0, active=True, time_to_divide=0, label_intensity=0.0, creation_time=0):
        self.cell_type = cell_type
        self.state = state
        self.age = age
        self.unique_id = unique_id
        self.params = params
        self.lt_rng = lt_rng
        self.st_rng = st_rng
        self.mpp_rng = mpp_rng
        self.parent_id = parent_id
        self.division_count = division_count
        self.time_to_divide = time_to_divide
        self.label_intensity = label_intensity
        self.creation_time = creation_time
        self.last_division_fate = None
        self.end_reason = None
        self.first_division = True

    @property
    def is_labeled(self):
        return self.label_intensity > self.params["LABEL_THRESHOLD"]

def initialize_cells(initial_lt_hsc_count, initial_st_hsc_count, params, lt_rng, st_rng, mpp_rng, initial_lt_quiescent, initial_st_quiescent):
    global cell_id_counter
    cells = []

    # Initialize active LT-HSCs
    for i in range(initial_lt_hsc_count):
        time_to_divide = max(2, round(lt_rng.normal(params["LT_HSC_TIME_MEAN"], params["LT_HSC_TIME_STD"])))
        uid = cell_id_counter
        cell_id_counter += 1
        label_intensity = 1.0 if lt_rng.random() < params["LABELING_FRACTION"] else 0.0
        cells.append(Cell(
            cell_type="LT-HSC",
            state="Non-quiescent Active",
            age=0,
            unique_id=uid,
            params=params,
            lt_rng=lt_rng,
            st_rng=st_rng,
            mpp_rng=mpp_rng,
            time_to_divide=time_to_divide,
            label_intensity=label_intensity
        ))

    # Initialize active ST-HSCs
    for i in range(initial_st_hsc_count):
        time_to_divide = max(2, round(st_rng.normal(params["ST_HSC_TIME_MEAN"], params["ST_HSC_TIME_STD"])))
        uid = cell_id_counter
        cell_id_counter += 1
        cells.append(Cell(
            cell_type="ST-HSC",
            state="Non-quiescent Active",
            age=0,
            unique_id=uid,
            params=params,
            lt_rng=lt_rng,
            st_rng=st_rng,
            mpp_rng=mpp_rng,
            time_to_divide=time_to_divide,
            label_intensity=0.0
        ))

    # Initialize quiescent LT-HSCs
    for i in range(initial_lt_quiescent):
        uid = cell_id_counter
        cell_id_counter += 1
        cells.append(Cell(
            cell_type="LT-HSC",
            state="Quiescent",
            age=0,
            unique_id=uid,
            params=params,
            lt_rng=lt_rng,
            st_rng=st_rng,
            mpp_rng=mpp_rng,
            time_to_divide=0,
            label_intensity=0.0
        ))

    # Initialize quiescent ST-HSCs
    for i in range(initial_st_quiescent):
        uid = cell_id_counter
        cell_id_counter += 1
        cells.append(Cell(
            cell_type="ST-HSC",
            state="Quiescent",
            age=0,
            unique_id=uid,
            params=params,
            lt_rng=lt_rng,
            st_rng=st_rng,
            mpp_rng=mpp_rng,
            time_to_divide=0,
            label_intensity=0.0
        ))

    return cells

# The rest of the code remains unchanged, except that any references to `LT_HSC_DAILY_DIVISION_RATE` and `ST_HSC_DAILY_DIVISION_RATE` have been removed.
def create_cell(cell_type, state, age, params, lt_rng, st_rng, mpp_rng, parent_id, parent_label_intensity, current_time):
    """
    Create a new cell with the specified attributes.
    Args:
        cell_type (str): The type of the cell (e.g., "LT-HSC", "ST-HSC", "MPP").
        state (str): The state of the cell (e.g., "Non-quiescent Active", "Non-quiescent Inactive", "Quiescent", "Apoptotic").
        age (int): The age of the cell in days.
        params (dict): Model parameters.
        lt_rng (np.random.RandomState): RNG for LT-HSC decisions.
        st_rng (np.random.RandomState): RNG for ST-HSC decisions.
        mpp_rng (np.random.RandomState): RNG for MPP decisions.
        parent_id (int): The unique ID of the parent cell.
        parent_label_intensity (float): The label intensity of the parent cell.
        current_time (int): The current time step in the simulation.
    Returns:
        Cell: A new Cell object with the specified attributes.
    """
    global cell_id_counter  # Use the global counter for unique IDs

    # Determine cell-specific attributes
    if cell_type == "LT-HSC":
        rng = lt_rng
        time_to_divide = lt_rng.normal(params["LT_HSC_TIME_MEAN"], params["LT_HSC_TIME_STD"])
        time_to_divide = max(2, round(time_to_divide))  # Ensure a minimum time to divide of 2 days
    elif cell_type == "ST-HSC":
        rng = st_rng
        time_to_divide = st_rng.normal(params["ST_HSC_TIME_MEAN"], params["ST_HSC_TIME_STD"])
        time_to_divide = max(2, round(time_to_divide))  # Ensure a minimum time to divide of 2 days
    else:  # MPP
        rng = mpp_rng
        time_to_divide = 0  # MPPs don't divide in the current model

    # Assign a unique ID to the new cell
    uid = cell_id_counter
    cell_id_counter += 1  # Increment the global counter

    # Calculate label intensity
    label_intensity = parent_label_intensity / 2.0  # Halve the parent's label intensity
    if cell_type == "LT-HSC" and current_time < params["LABELING_PERIOD"] and label_intensity == 0:
        # Apply additional labeling logic during the labeling period
        if lt_rng.random() < params["LABELING_PROB"]:
            label_intensity = 1.0

    # Create and return the new cell
    return Cell(
        cell_type=cell_type,
        state=state,
        age=age,
        unique_id=uid,
        params=params,
        lt_rng=lt_rng,
        st_rng=st_rng,
        mpp_rng=mpp_rng,
        parent_id=parent_id,
        time_to_divide=time_to_divide,
        label_intensity=label_intensity,
        creation_time=current_time
    )

def check_apoptosis(cell):
    """
    Determine if a cell undergoes apoptosis based on its state and type.
    Args:
        cell (Cell): The cell to check.
    Returns:
        bool: True if the cell undergoes apoptosis, False otherwise.
    """
    if cell.state == "Non-quiescent Active":
        if cell.cell_type == "LT-HSC":
            return cell.lt_rng.random() < cell.params["PAA"]
        elif cell.cell_type == "ST-HSC":
            return cell.st_rng.random() < cell.params["PAB"]
        else:  # MPP
            return False  # MPPs don't undergo apoptosis in the current model
    elif cell.state == "Non-quiescent Inactive":
        # Inactive cells do not undergo apoptosis in the current model
        return False
    elif cell.state == "Quiescent":
        if cell.cell_type == "LT-HSC":
            return cell.lt_rng.random() < cell.params["PAQA"]
        elif cell.cell_type == "ST-HSC":
            return cell.st_rng.random() < cell.params["PAQB"]
        else:  # MPP
            return False
    return False

def check_quiescence(cell):
    """
    Determine if a cell enters quiescence based on its type and a random probability.
    Args:
        cell (Cell): The cell to check.
    Returns:
        bool: True if the cell enters quiescence, False otherwise.
    """
    if cell.cell_type == "LT-HSC":
        # LT-HSCs enter quiescence with probability PQA
        return cell.lt_rng.random() < cell.params["PQA"]
    elif cell.cell_type == "ST-HSC":
        # ST-HSCs enter quiescence with probability PQB
        return cell.st_rng.random() < cell.params["PQB"]
    # MPPs do not enter quiescence in the current model
    return False

def perform_division(cell, cells, current_time, division_events):
    """
    Perform division for a given cell and update the cell population.
    Args:
        cell (Cell): The cell undergoing division.
        cells (list): The list of all cells in the simulation.
        current_time (int): The current time step in the simulation.
    Returns:
        str: The fate of the cell (e.g., "symmetric_self_renewal", "asymmetric_division").
    """
    fate = None  # Initialize the fate of the cell

    # Handle LT-HSC division logic: s determine the fate of the mother cell during division and  create new daughter cells
    if cell.cell_type == "LT-HSC":
        rand_fate = cell.lt_rng.random()  # Generate a random number for fate decision
        if rand_fate < cell.params["P1A"]:
            # Symmetric Self-Renewal: LT-HSC -> 2 LT-HSCs
            cells.append(create_cell("LT-HSC", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            fate = "symmetric_self_renewal"
        elif rand_fate < (cell.params["P1A"] + cell.params["P2A"]):
            # Asymmetric Division: LT-HSC -> LT-HSC + ST-HSC
            cells.append(create_cell("ST-HSC", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            fate = "asymmetric_division"
        elif rand_fate < (cell.params["P1A"] + cell.params["P2A"] + cell.params["P3A"]):
            # Symmetric Differentiation: LT-HSC -> 1 ST-HSC + death
            cells.append(create_cell("ST-HSC", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            cell.state = "Apoptotic"  # The parent LT-HSC dies
            fate = "symmetric_differentiation"
        else:
            # Full Differentiation: LT-HSC -> 2 ST-HSCs + death
            cells.append(create_cell("ST-HSC", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            cells.append(create_cell("ST-HSC", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            cell.state = "Apoptotic"  # The parent LT-HSC dies
            fate = "full_differentiation"

    # Handle ST-HSC division logic
    elif cell.cell_type == "ST-HSC":
        rand_fate = cell.st_rng.random()  # Generate a random number for fate decision
        if rand_fate < cell.params["P1B"]:
            # Symmetric Self-Renewal: ST-HSC -> 2 ST-HSCs
            cells.append(create_cell("ST-HSC", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            fate = "symmetric_self_renewal"
        elif rand_fate < (cell.params["P1B"] + cell.params["P2B"]):
            # Asymmetric Division: ST-HSC -> ST-HSC + MPP
            cells.append(create_cell("MPP", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            fate = "asymmetric_division"
        elif rand_fate < (cell.params["P1B"] + cell.params["P2B"] + cell.params["P3B"]):
            # Symmetric Differentiation: ST-HSC -> 1 MPP + death
            cells.append(create_cell("MPP", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            cell.state = "Apoptotic"  # The parent ST-HSC dies
            fate = "symmetric_differentiation"
        else:
            # Full Differentiation: ST-HSC -> 2 MPPs + death
            cells.append(create_cell("MPP", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            cells.append(create_cell("MPP", "Non-quiescent Active", 0, cell.params, cell.lt_rng, cell.st_rng, cell.mpp_rng, cell.unique_id, cell.label_intensity, current_time))
            cell.state = "Apoptotic"  # The parent ST-HSC dies
            fate = "full_differentiation"

    # MPPs don't divide in the current model


    # Record the division event
    daughter_types = [daughter.cell_type for daughter in cells if daughter.parent_id == cell.unique_id]
    division_events.append({
        "time": current_time,
        "parent_id": cell.unique_id,
        "parent_type": cell.cell_type,
        "parent_age": cell.age,
        "daughter_types": daughter_types,
        "fate": fate
    })

    # Halve the mother's label intensity
    cell.label_intensity /= 2.0  # Update the mother's label intensity

    # Store the fate of the division
    cell.last_division_fate = fate

    # Increment the division count
    cell.division_count += 1

    # Check if the cell has reached its division threshold
    division_threshold = cell.params["LT_HSC_DIVISION_THRESHOLD"] if cell.cell_type == "LT-HSC" else cell.params["ST_HSC_DIVISION_THRESHOLD"]
    if cell.division_count >= division_threshold:
        cell.state = "Non-quiescent Inactive"  # Transition to inactive state

   # Update the time to divide for the parent cell
    if cell.cell_type in ["LT-HSC", "ST-HSC"]:
        rng = cell.lt_rng if cell.cell_type == "LT-HSC" else cell.st_rng
        time_mean = cell.params[f"{cell.cell_type.replace('-', '_')}_TIME_MEAN"]
        time_std = cell.params[f"{cell.cell_type.replace('-', '_')}_TIME_STD"]
        cell.time_to_divide = max(2, round(rng.normal(time_mean, time_std)))

    # Return the fate for tracking
    return fate

def update_cell(cell, cells, current_time, division_events):
    cell.time_to_divide -= 1

    # Apply labeling logic with delay
    if (
        cell.cell_type == "LT-HSC" and
        current_time >= cell.params["LABELING_DELAY"] and
        current_time < cell.params["LABELING_DELAY"] + cell.params["LABELING_PERIOD"] and
        cell.label_intensity == 0
    ):
        if cell.state in ["Non-quiescent Active", "Quiescent"]:
            if cell.lt_rng.random() < cell.params["LABELING_PROB"]:
                cell.label_intensity = 1.0

    # Handle active cells
    if cell.state == "Non-quiescent Active":
        if cell.cell_type in ["LT-HSC", "ST-HSC"]:
            if check_quiescence(cell):
                cell.state = "Quiescent"
            elif check_apoptosis(cell):
                cell.state = "Apoptotic"
            elif cell.time_to_divide <= 0:
                perform_division(cell, cells, current_time, division_events)

    elif cell.state == "Quiescent":
        if check_apoptosis(cell):
            cell.state = "Apoptotic"
        else:
            exit_prob = cell.params["P_EXIT_QUIESCENCE_A"] if cell.cell_type == "LT-HSC" else cell.params["P_EXIT_QUIESCENCE_B"]
            rng = cell.lt_rng if cell.cell_type == "LT-HSC" else cell.st_rng
            if rng.random() < exit_prob:
                cell.state = "Non-quiescent Active"
                time_mean = cell.params["LT_HSC_TIME_MEAN"] if cell.cell_type == "LT-HSC" else cell.params["ST_HSC_TIME_MEAN"]
                time_std = cell.params["LT_HSC_TIME_STD"] if cell.cell_type == "LT-HSC" else cell.params["ST_HSC_TIME_STD"]
                cell.time_to_divide = max(2, round(rng.normal(time_mean, time_std)))

def simulate_hematopoiesis(initial_lt_hsc_count, initial_st_hsc_count, params, time_steps, sample_days, initial_lt_quiescent, initial_st_quiescent, seed=None,initial_cells=None):
    """
    Simulate the hematopoiesis process over a given number of time steps. 
    Args:
        initial_lt_hsc_count (int): Initial number of LT-HSCs.
        initial_st_hsc_count (int): Initial number of ST-HSCs.
        params (dict): Model parameters.
        time_steps (int): Total number of time steps to simulate.
        sample_days (list): Days at which to sample cell counts.
        initial_lt_quiescent (int): Number of initial quiescent LT-HSCs
        initial_st_quiescent (int): Number of initial quiescent ST-HSCs
        seed (int): Random seed for reproducibility. Defaults to None. 
    Returns:
        tuple: A tuple containing:
            - counts (dict): Cell counts at sampled time points.
            - cells_history (list): History of removed cells.
            - division_events (list): History of cell division events.
            - apoptosis_events (list): History of apoptosis events.

"""
    # Create separate random number generators for LT-HSCs, ST-HSCs, and MPPs
    lt_rng = np.random.RandomState(seed)
    st_rng = np.random.RandomState(seed + 1 if seed is not None else None)
    mpp_rng = np.random.RandomState(seed + 2 if seed is not None else None)
    

    # Also set the random module's seed for compatibility
    # The seeds above set the random stream for numpy, but we also want to control the seed for python's random module
    if seed is not None:
        random.seed(seed)

    # Use pre-stabilized cells if provided
    if initial_cells is not None:
        cells = initial_cells

        # Add labeling logic for pre-stabilized cells
        for cell in cells:
            if cell.cell_type == "LT-HSC" and cell.state in ["Non-quiescent Active", "Quiescent"]:
                # Apply labeling based on the LABELING_FRACTION and LABELING_PROB
                if lt_rng.random() < params["LABELING_FRACTION"]:
                    cell.label_intensity = 1.0
    else:
        # Initialize the initial population of cells
        cells = initialize_cells(initial_lt_hsc_count, initial_st_hsc_count, params, lt_rng, st_rng, mpp_rng, initial_lt_quiescent, initial_st_quiescent)


    
    # Initialize tracking structures
    ## there has to be a better way to do this history stuff. But first I need to decide what we will be reporting.
    cells_history = []  # Will store complete cell history for removed cells
    division_events = []  # Will store division events for analysis
    apoptosis_events = []  # Will store apoptosis events for analysis
    
    # Initialize the initial population of cells
    cells = initialize_cells(initial_lt_hsc_count, initial_st_hsc_count, params, lt_rng, st_rng, mpp_rng, initial_lt_quiescent, initial_st_quiescent)    
    # Add creation_time and initialize last_division_fate for each initial cell
    for cell in cells:
        cell.creation_time = 0
        cell.last_division_fate = None  # Initialize this attribute for tracking division fates
    
    # Initialize a dictionary to store cell counts at sampled time points
    counts = {
        "LT-HSC_active_labeled": [],
        "LT-HSC_active_unlabeled": [],
        "LT-HSC_quiescent_labeled": [],
        "LT-HSC_quiescent_unlabeled": [],
        "LT-HSC_nonquiescent_inactive": [], 
        "ST-HSC_active_labeled": [],
        "ST-HSC_active_unlabeled": [],
        "ST-HSC_quiescent_labeled": [],
        "ST-HSC_quiescent_unlabeled": [],
        "ST-HSC_nonquiescent_inactive": [],  
        "MPP_total": [],
        "LT-HSC_removed": [],
        "ST-HSC_removed": [],
    }
    
    # Convert sample_days to integer time steps
    sample_times = [int(day) for day in sample_days]
    
    # Main simulation loop
    for t in range(time_steps):

        new_cells = []  # List to store newly created cells
        removed_cells = []  # List to store cells that will be removed (apoptotic cells)
        
        # Process each cell in the current population
        for cell in cells:
            if cell.state != "Apoptotic":  # Skip cells that are already apoptotic
                old_state = cell.state  # Track the cell's state before updating
                old_division_count = cell.division_count  # Track the division count before updating
                
                # Update the cell's state and handle division or apoptosis
                update_cell(cell, new_cells, t, division_events)
                
                # Track direct apoptosis (not caused by division so it doesn't check cells that just divided)
                if cell.state == "Apoptotic" and old_state != "Apoptotic" and cell.division_count == old_division_count:
                    apoptosis_events.append({
                        "time": t,
                        "cell_id": cell.unique_id,
                        "cell_type": cell.cell_type,
                        "cell_age": cell.age,
                        "reason": "direct_apoptosis"  # Reason for apoptosis
                    })
                
                # Track division events
                if cell.division_count > old_division_count:
                    # Get daughter cells created during this division
                    daughter_cells = [new_cell for new_cell in new_cells if new_cell.parent_id == cell.unique_id]
                    daughter_types = [daughter.cell_type for daughter in daughter_cells]
                    
                    division_events.append({
                        "time": t,
                        "parent_id": cell.unique_id,
                        "parent_type": cell.cell_type,
                        "parent_age": cell.age,
                        "daughter_types": daughter_types,
                        "fate": cell.last_division_fate
                    })
            
            # Increment the age of the cell
            cell.age += 1
            
            # Identify cells that will be removed (apoptotic cells)
            if cell.state == "Apoptotic":
            # Set the reason for apoptosis based on the division fate or default to "direct_apoptosis"
                if cell.last_division_fate in ["symmetric_differentiation", "full_differentiation"]:
                    cell.end_reason = "apoptosis_due_to_differentiation"
                else:
                    cell.end_reason = "direct_apoptosis"
                
                # Add the cell to the removed_cells list
                removed_cells.append(cell)

        # Record history for removed cells
        for cell in removed_cells:
            cells_history.append({
                "unique_id": cell.unique_id,
                "cell_type": cell.cell_type,
                "start_time": cell.creation_time,
                "end_time": t,
                "end_reason": cell.end_reason if hasattr(cell, "end_reason") else "unknown",
                "parent_id": cell.parent_id,
                "division_count": cell.division_count  
            })
        
        # Add new cells to the population and remove apoptotic cells
        cells.extend(new_cells)
        cells = [cell for cell in cells if cell.state != "Apoptotic"]
        
        # Sample cell counts at specified time points
        if t in sample_times:
            # LT-HSC counts
            lt_hsc_active_labeled = sum(1 for cell in cells if cell.cell_type == "LT-HSC" and cell.state == "Non-quiescent Active" and cell.is_labeled)
            lt_hsc_active_unlabeled = sum(1 for cell in cells if cell.cell_type == "LT-HSC" and cell.state == "Non-quiescent Active" and not cell.is_labeled)
            lt_hsc_quiescent_labeled = sum(1 for cell in cells if cell.cell_type == "LT-HSC" and cell.state == "Quiescent" and cell.is_labeled)
            lt_hsc_quiescent_unlabeled = sum(1 for cell in cells if cell.cell_type == "LT-HSC" and cell.state == "Quiescent" and not cell.is_labeled)
            lt_hsc_nonquiescent_inactive = sum(1 for cell in cells if cell.cell_type == "LT-HSC" and cell.state == "Non-quiescent Inactive") 


            # ST-HSC counts
            st_hsc_active_labeled = sum(1 for cell in cells if cell.cell_type == "ST-HSC" and cell.state == "Non-quiescent Active" and cell.is_labeled)
            st_hsc_active_unlabeled = sum(1 for cell in cells if cell.cell_type == "ST-HSC" and cell.state == "Non-quiescent Active" and not cell.is_labeled)
            st_hsc_quiescent_labeled = sum(1 for cell in cells if cell.cell_type == "ST-HSC" and cell.state == "Quiescent" and cell.is_labeled)
            st_hsc_quiescent_unlabeled = sum(1 for cell in cells if cell.cell_type == "ST-HSC" and cell.state == "Quiescent" and not cell.is_labeled)
            st_hsc_nonquiescent_inactive = sum(1 for cell in cells if cell.cell_type == "ST-HSC" and cell.state == "Non-quiescent Inactive") 
            

            # MPP counts
            mpp_total = sum(1 for cell in cells if cell.cell_type == "MPP")

            # Removed LT-HSCs and ST-HSCs
            lt_hsc_removed = sum(1 for cell in removed_cells if cell.cell_type == "LT-HSC")
            st_hsc_removed = sum(1 for cell in removed_cells if cell.cell_type == "ST-HSC")

            # Append counts to the dictionary
            counts["LT-HSC_active_labeled"].append(lt_hsc_active_labeled)
            counts["LT-HSC_active_unlabeled"].append(lt_hsc_active_unlabeled)
            counts["LT-HSC_quiescent_labeled"].append(lt_hsc_quiescent_labeled)
            counts["LT-HSC_quiescent_unlabeled"].append(lt_hsc_quiescent_unlabeled)
            counts["LT-HSC_nonquiescent_inactive"].append(lt_hsc_nonquiescent_inactive)  
            counts["ST-HSC_active_labeled"].append(st_hsc_active_labeled)
            counts["ST-HSC_active_unlabeled"].append(st_hsc_active_unlabeled)
            counts["ST-HSC_quiescent_labeled"].append(st_hsc_quiescent_labeled)
            counts["ST-HSC_quiescent_unlabeled"].append(st_hsc_quiescent_unlabeled)
            counts["ST-HSC_nonquiescent_inactive"].append(st_hsc_nonquiescent_inactive) 
            counts["MPP_total"].append(mpp_total)
            counts["LT-HSC_removed"].append(lt_hsc_removed)  
            counts["ST-HSC_removed"].append(st_hsc_removed)  


    # Return both the counts and the tracking data
    return counts, cells, cells_history, division_events, apoptosis_events

# ## Example usage

# ## Setup initial parameters
# initial_lt_hsc_count = 5000
# initial_lt_quiescent = 15000
# initial_st_hsc_count = 5
# initial_st_quiescent = 5
# time_steps = 7 * 40 #(50 weeks)
# sample_days = list(range(0, time_steps, 7))  # If time steps is 7*n, this will produce n samples
# seed = 42

# ## output
# counts, cells, cells_history, division_events, apoptosis_event = simulate_hematopoiesis(
#     initial_lt_hsc_count, 
#     initial_st_hsc_count, 
#     params, 
#     time_steps, 
#     sample_days,
#     initial_lt_quiescent, 
#     initial_st_quiescent,  
#     seed=seed
# )



# # Extract the number of LT-HSC_active_unlabeled cells over time
# lt_hsc_labeled = np.array(counts["LT-HSC_active_labeled"])+ np.array(counts["LT-HSC_quiescent_labeled"])

# # Print the results
# print("LT-HSC labeled over time (sample_days):")
# for day, count in zip(sample_days, lt_hsc_labeled):
#     print(f"Day {day}: {count}")

# lt_hsc_unlabeled = np.array(counts["LT-HSC_active_unlabeled"])+ np.array(counts["LT-HSC_quiescent_unlabeled"])
# frac_lt_hsc_labeled = lt_hsc_labeled / (lt_hsc_labeled + lt_hsc_unlabeled)
# print("Fraction of LT-HSC labeled over time (sample_days):")
# for day, frac in zip(sample_days, frac_lt_hsc_labeled):
#     print(f"Day {day}: {frac:.2f}")

# # Extract the number of LT-HSC_active_unlabeled cells over time
# st_hsc_labeled = np.array(counts["ST-HSC_active_labeled"])+ np.array(counts["ST-HSC_quiescent_labeled"])
# frac_st_hsc_labeled = st_hsc_labeled / lt_hsc_labeled
# # Print the results
# print("ST-HSC labeled over time (sample_days):")
# for day, count in zip(sample_days, frac_st_hsc_labeled):
#     print(f"Day {day}: {count}")