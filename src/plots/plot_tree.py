import numpy as np
import random

# Default parameters for the hematopoiesis model
params = {
    # Division thresholds: Maximum number of divisions before a cell becomes inactive
    "LT_HSC_DIVISION_THRESHOLD": 5,  # Long-term HSC division threshold
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
    "P_EXIT_QUIESCENCE_B": 0.15,   # ST-HSCs exiting quiescence

    # Division fate probabilities for LT-HSCs
    "P1A": 0.6,  # Symmetric Self-Renewal: LT-HSC -> 2 LT-HSCs
    "P2A": 0.096,  # Asymmetric Division: LT-HSC -> LT-HSC + ST-HSC
    "P3A": 0.27968000000000004,  # Symmetric Differentiation: LT-HSC -> 1 ST-HSC + death
    "P4A": 0.024320000000000005,  # Full Differentiation: LT-HSC -> 2 ST-HSCs + death

    # Division fate probabilities for ST-HSCs
    "P1B": 0.5,  # Symmetric Self-Renewal: ST-HSC -> 2 ST-HSCs
    "P2B": 0.15,    # Asymmetric Division: ST-HSC -> ST-HSC + MPP
    "P3B": 0.1,    # Symmetric Differentiation: ST-HSC -> 1 MPP + death
    "P4B": 0.25,  # Full Differentiation: ST-HSC -> 2 MPPs + death

    # Division timing parameters for LT-HSCs
    "LT_HSC_TIME_MEAN": 40,  # Mean time to divide (days)
    "LT_HSC_TIME_STD":12.5,   # Standard deviation of division time (days)
    "LT_HSC_DAILY_DIVISION_RATE": 0.1,  # 1.6% daily division rate for LT-HSCs    

    # Division timing parameters for ST-HSCs
    "ST_HSC_TIME_MEAN": 12,  # Mean time to divide (days)
    "ST_HSC_TIME_STD": 5.5,   # Standard deviation of division time (days)
    "ST_HSC_DAILY_DIVISION_RATE": 0.28,  # 6% daily division rate for ST-HSCs

    # Labeling parameters
    "LABEL_THRESHOLD": 0.06,  # Threshold for determining if a cell is labeled
    "LABELING_PERIOD": 2,       # Time period during which labeling occurs (days)
    "LABELING_PROB": 0.005     # Probability of labeling a cell during the labeling period
}


# Global counter for unique cell IDs
cell_id_counter = 0

class Cell:
    def __init__(self, cell_type, state, age, unique_id, params, lt_rng, st_rng, mpp_rng, parent_id=None, division_count=0, active=True, time_to_divide=0, label_intensity=0.0, creation_time=0):
        """
        Initialize a Cell object.
        """
        self.cell_type = cell_type  # (str): The type of the cell (e.g., "LT-HSC", "ST-HSC", "MPP").
        self.state = state  # (str): The state of the cell (e.g., "Non-quiescent Active", "Non-quiescent Inactive", "Quiescent", "Apoptotic").
        self.age = age  # (int): The age of the cell in days.
        self.unique_id = unique_id  # (int): A unique identifier for the cell.
        self.params = params  # (dict): Model parameters.
        self.lt_rng = lt_rng  # (np.random.RandomState): RNG for LT-HSC decisions.
        self.st_rng = st_rng  # (np.random.RandomState): RNG for ST-HSC decisions.
        self.mpp_rng = mpp_rng  # (np.random.RandomState): RNG for MPP decisions.
        self.parent_id = parent_id  # (int): The unique ID of the parent cell. Defaults to None for initial cells.
        self.division_count = division_count  #(int): The number of divisions the cell has undergone. Defaults to 0 for iniital cells. 
        self.time_to_divide = time_to_divide  # (int): Time remaining until the cell divides. Defaults to 0.
        self.label_intensity = label_intensity  # (float): The intensity of the label (e.g., for pulse labeling). Defaults to 0.0 for unlabeled.
        self.creation_time = creation_time  # (int): The time step when the cell was created. Defaults to 0 for initial cells. 
        self.last_division_fate = None  # (str) Track the last division fate (e.g., symmetric_self_renewal, asymmetric_division, symmetric_differentiation, full_differentiation)
        self.end_reason = None  # Reason for apoptosis or removal
    @property
    def is_labeled(self):
        """
        Determine if the cell is labeled based on its label intensity.
        Returns:
            bool: True if the cell's label intensity exceeds the labeling threshold, False otherwise.
        Note:
            This property is necessary to simplify the logic for determining whether a cell is labeled.
            It abstracts the comparison between the cell's label intensity and the labeling threshold,
            making the code more readable and reusable wherever this check is needed.
        """
        return self.label_intensity > self.params["LABEL_THRESHOLD"]
    

def initialize_cells(initial_lt_hsc_count, initial_st_hsc_count, params, lt_rng, st_rng, mpp_rng, label_fraction=0.1):
    """
    Initialize the populations of LT-HSCs and ST-HSCs.
    Args:
        initial_lt_hsc_count (int): Number of initial LT-HSCs to create.
        initial_st_hsc_count (int): Number of initial ST-HSCs to create.
        params (dict): Model parameters.
        lt_rng (np.random.RandomState): RNG for LT-HSC decisions.
        st_rng (np.random.RandomState): RNG for ST-HSC decisions.
        mpp_rng (np.random.RandomState): RNG for MPP decisions (not used here).
        label_fraction (float): Fraction of LT-HSCs to label initially. Defaults to 0.1.
    Returns:
        list: A list of initialized Cell objects (LT-HSCs and ST-HSCs).
    """

    global cell_id_counter  # Use the global counter
    cells = []  # List to store the initialized cells

    # Initialize LT-HSCs
    for i in range(initial_lt_hsc_count):
        # Determine the time to divide for the LT-HSC
        time_to_divide = lt_rng.normal(params["LT_HSC_TIME_MEAN"], params["LT_HSC_TIME_STD"])
        time_to_divide = max(2, round(time_to_divide))  # Ensure a minimum time to divide of 2 days

        # Generate a unique ID for the cell
        uid = cell_id_counter  # Assign the current counter value as the unique ID
        cell_id_counter += 1  # Increment the counter

        # Determine the label intensity for the cell
        label_intensity = 1.0 if lt_rng.random() < label_fraction else 0.0

        # Create the LT-HSC and add it to the list
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

    # Initialize ST-HSCs
    for i in range(initial_st_hsc_count):
        # Determine the time to divide for the ST-HSC
        time_to_divide = st_rng.normal(params["ST_HSC_TIME_MEAN"], params["ST_HSC_TIME_STD"])
        time_to_divide = max(2, round(time_to_divide))  # Ensure a minimum time to divide of 2 days

        # Generate a unique ID for the cell
        uid = cell_id_counter  # Assign the current counter value as the unique ID
        cell_id_counter += 1  # Increment the counter

        # ST-HSCs are not labeled initially, so label_intensity is 0.0
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

    return cells  # Return the list of initialized cells


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
    """
    Update the state of a cell during a single time step.
    Args:
        cell (Cell): The cell to update.
        cells (list): The list of all cells in the simulation (used to add new cells from division).
        current_time (int): The current time step in the simulation.
    """

    # Decrease the time remaining until the cell divides
    cell.time_to_divide -= 1

    # Handle active cells
    if cell.state == "Non-quiescent Active":
        # Check if the cell should enter quiescence
        if cell.cell_type in ["LT-HSC", "ST-HSC"]:
            if check_quiescence(cell):
                # Transition the cell to the quiescent state
                cell.state = "Quiescent"
            elif check_apoptosis(cell):
                # Transition the cell to the apoptotic state if apoptosis occurs
                cell.state = "Apoptotic"
            elif cell.time_to_divide <= 0:
                # If the cell is ready to divide, perform division
                fate = perform_division(cell, cells, current_time, division_events)

    # Handle quiescent cells
    elif cell.state == "Quiescent":
        # Check if the cell undergoes apoptosis while quiescent
        if check_apoptosis(cell):
            cell.state = "Apoptotic"
        else:
            # Determine the probability of exiting quiescence based on the cell type
            exit_prob = cell.params["P_EXIT_QUIESCENCE_A"] if cell.cell_type == "LT-HSC" else cell.params["P_EXIT_QUIESCENCE_B"]
            rng = cell.lt_rng if cell.cell_type == "LT-HSC" else cell.st_rng
            if rng.random() < exit_prob:
                # Transition the cell back to the active state
                cell.state = "Non-quiescent Active"
                # Reset the time to divide to a new random value
                time_mean = cell.params["LT_HSC_TIME_MEAN"] if cell.cell_type == "LT-HSC" else cell.params["ST_HSC_TIME_MEAN"]
                time_std = cell.params["LT_HSC_TIME_STD"] if cell.cell_type == "LT-HSC" else cell.params["ST_HSC_TIME_STD"]
                rng = cell.lt_rng if cell.cell_type == "LT-HSC" else cell.st_rng
                cell.time_to_divide = max(2, round(rng.normal(time_mean, time_std)))


def simulate_hematopoiesis(initial_lt_hsc_count, initial_st_hsc_count, params, time_steps, sample_days, label_fraction=0.1, seed=None):
    """
    Simulate the hematopoiesis process over a given number of time steps. 
    Args:
        initial_lt_hsc_count (int): Initial number of LT-HSCs.
        initial_st_hsc_count (int): Initial number of ST-HSCs.
        params (dict): Model parameters.
        time_steps (int): Total number of time steps to simulate.
        sample_days (list): Days at which to sample cell counts.
        label_fraction (float): Fraction of LT-HSCs to label initially. Defaults to 0.1.
        seed (int): Random seed for reproducibility. Defaults to None. 
    Returns:
        tuple: A tuple containing:
            - counts (dict): Cell counts at sampled time points.
            - cells_history (list): History of removed cells.
            - division_events (list): History of cell division events.
    """
    # Create separate random number generators for LT-HSCs, ST-HSCs, and MPPs
    lt_rng = np.random.RandomState(seed)
    st_rng = np.random.RandomState(seed + 1 if seed is not None else None)
    mpp_rng = np.random.RandomState(seed + 2 if seed is not None else None)
    
    # Also set the random module's seed for compatibility
    # The seeds above set the random stream for numpy, but we also want to control the seed for python's random module
    if seed is not None:
        random.seed(seed)
    
    # Initialize tracking structures
    ## there has to be a better way to do this history stuff. But first I need to decide what we will be reporting.
    cells_history = []  # Will store complete cell history for removed cells
    division_events = []  # Will store division events for analysis
    apoptosis_events = []  # Will store apoptosis events for analysis
    
    # Initialize the initial population of cells
    cells = initialize_cells(initial_lt_hsc_count, initial_st_hsc_count, params, lt_rng, st_rng, mpp_rng, label_fraction)
    
    # Add creation_time and initialize last_division_fate for each initial cell
    for cell in cells:
        cell.creation_time = 0
        cell.last_division_fate = None  # Initialize this attribute for tracking division fates
    
    # Initialize a dictionary to store cell counts at sampled time points
    counts = {
        "LT-HSC_labeled": [],
        "LT-HSC_unlabeled": [],
        "ST-HSC_labeled": [],
        "ST-HSC_unlabeled": [],
        "MPP_labeled": [],
        "MPP_unlabeled": []
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
                "state": "Removed" 
            })
        
        # Add new cells to the population and remove apoptotic cells
        cells.extend(new_cells)
        cells = [cell for cell in cells if cell.state != "Apoptotic"]

        
        # Sample cell counts at specified time points
        if t in sample_times:
            lt_hsc_labeled = sum(1 for cell in cells if cell.cell_type == "LT-HSC" and cell.is_labeled)
            lt_hsc_total = sum(1 for cell in cells if cell.cell_type == "LT-HSC")
            lt_hsc_unlabeled = lt_hsc_total - lt_hsc_labeled
            
            st_hsc_labeled = sum(1 for cell in cells if cell.cell_type == "ST-HSC" and cell.is_labeled)
            st_hsc_total = sum(1 for cell in cells if cell.cell_type == "ST-HSC")
            st_hsc_unlabeled = st_hsc_total - st_hsc_labeled
            
            mpp_labeled = sum(1 for cell in cells if cell.cell_type == "MPP" and cell.is_labeled)
            mpp_total = sum(1 for cell in cells if cell.cell_type == "MPP")
            mpp_unlabeled = mpp_total - mpp_labeled
            
            counts["LT-HSC_labeled"].append(lt_hsc_labeled)
            counts["LT-HSC_unlabeled"].append(lt_hsc_unlabeled)
            counts["ST-HSC_labeled"].append(st_hsc_labeled)
            counts["ST-HSC_unlabeled"].append(st_hsc_unlabeled)
            counts["MPP_labeled"].append(mpp_labeled)
            counts["MPP_unlabeled"].append(mpp_unlabeled)
    
    # Return both the counts and the tracking data
    return counts, cells_history, division_events, apoptosis_events, cells

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx

# Run the simulation
initial_lt_hsc_count = 2*250
initial_st_hsc_count = 3*2*250
initial_lt_quiescent = 2*750
initial_st_quiescent = 3*2*250
time_steps = 7 * 100
sample_days = list(range(0, time_steps, 7))  # If time steps is 7*n, this will produce n samples
label_fraction = 0.001
seed = 42

counts, cells_history, division_events, apoptosis_events, cells = simulate_hematopoiesis(
    initial_lt_hsc_count,
    initial_st_hsc_count,
    params,
    time_steps,
    sample_days,
    label_fraction=label_fraction,
    seed=seed
)

# Pick a random initial LT-HSC
initial_cell_id = 84

# Trace the lineage of the selected cell, including active and quiescent descendants
def trace_lineage(cell_id, cells, cells_history):
    """
    Trace the lineage of a given cell, including active, quiescent, and removed descendants.
    Args:
        cell_id (int): The unique ID of the cell to trace.
        cells (list): The current population of cells.
        cells_history (list): The history of removed cells.
    Returns:
        list: A list of cells in the lineage, including the selected cell and all its descendants.
    """
    lineage = []
    stack = [cell_id]  # Start with the selected cell ID

    # Combine current cells and cells_history for full lineage tracking
    all_cells = [
        {
            "unique_id": cell.unique_id,
            "parent_id": cell.parent_id,
            "cell_type": cell.cell_type,
            "state": cell.state,
        }
        for cell in cells
    ] + [
        {
            "unique_id": c["unique_id"],
            "parent_id": c["parent_id"],
            "cell_type": c["cell_type"],
            "state": c["state"],
        }
        for c in cells_history
    ]

    # Use a stack to perform depth-first search for descendants
    while stack:
        current_id = stack.pop()
        cell = next((c for c in all_cells if c["unique_id"] == current_id), None)
        if cell:
            lineage.append(cell)
            # Add all children of the current cell to the stack
            children = [c["unique_id"] for c in all_cells if c["parent_id"] == current_id]
            stack.extend(children)

    return lineage

# Trace the lineage of the selected cell
lineage = trace_lineage(initial_cell_id, cells, cells_history)


def plot_hierarchical_tree_with_progress(lineage, initial_cell_id):
    """
    Plot a hierarchical tree of the lineage with the initial cell at the top.
    Includes a progress counter during tree generation.
    Args:
        lineage (list): List of cells in the lineage.
        initial_cell_id (int): The unique ID of the initial cell.
    """
    import time

    G = nx.DiGraph()
    color_map = []
    node_shapes = []
    node_edge_colors = []

    # Add nodes and edges to the graph
    print("Generating hierarchical tree...")
    for i, cell in enumerate(lineage):
        G.add_node(cell["unique_id"], cell_type=cell["cell_type"])
        if cell["parent_id"] is not None:
            G.add_edge(cell["parent_id"], cell["unique_id"])

        # Assign colors and shapes based on cell type and state
        if cell["unique_id"] == initial_cell_id:
            node_edge_colors.append("black") 
            if cell.get("state") == "Non-quiescent Active":
                color_map.append("blue")
                node_shapes.append("o")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("dodgerblue")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("skyblue")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightblue")
                node_shapes.append("s")
        elif cell["cell_type"] == "LT-HSC":
            node_edge_colors.append("white") 
            if cell.get("state") == "Non-quiescent Active":
                color_map.append("blue")
                node_shapes.append("o")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("dodgerblue")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("skyblue")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightblue")
                node_shapes.append("s")
        elif cell["cell_type"] == "ST-HSC":
            node_edge_colors.append("white") 
            if cell.get("state") == "Non-quiescent Active":
                node_shapes.append("o")
                color_map.append("green")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("limegreen")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("palegreen")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightgreen")
                node_shapes.append("s")
        elif cell["cell_type"] == "MPP":
            node_edge_colors.append("white") 
            if cell.get("state") == "Non-quiescent Active":
                color_map.append("red")
                node_shapes.append("o")
            elif cell.get("state") == "Non-quiescent Inactive":
                color_map.append("orangered")
                node_shapes.append("d")
            elif cell.get("state") == "Quiescent":
                color_map.append("salmon")
                node_shapes.append("o")
            elif cell.get("state") == "Removed":
                color_map.append("lightcoral")
                node_shapes.append("s")

        # Print progress every 100 nodes
        if (i + 1) % 100 == 0 or i == len(lineage) - 1:
            print(f"Processed {i + 1}/{len(lineage)} nodes...")
            time.sleep(0.01)  # Simulate processing delay for visibility

    # Use a hierarchical layout (requires pygraphviz)
    pos = nx.nx_agraph.graphviz_layout(G, prog="dot")  # 'dot' creates a hierarchical layout

    # Plot the graph
    plt.figure(figsize=(12, 8))

    # Draw nodes with different shapes
    for shape in set(node_shapes):  # Iterate over unique shapes
        shape_nodes = [node for node, s in zip(G.nodes(), node_shapes) if s == shape]
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=shape_nodes,
            node_color=[color_map[i] for i, node in enumerate(G.nodes()) if node in shape_nodes],
            node_shape=shape,
            node_size=800,
            edgecolors=[node_edge_colors[i] for i, node in enumerate(G.nodes()) if node in shape_nodes],
            linewidths=2,  # Set the border width
        )

    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color="gray")
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Add a legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='LT-HSC Active'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='dodgerblue', markersize=10, label='LT-HSC Senescent'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10, label='LT-HSC Quiescent'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=10, label='LT-HSC Removed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='ST-HSC Active'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='limegreen', markersize=10, label='ST-HSC Senescent'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='palegreen', markersize=10, label='ST-HSC Quiescent'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightgreen', markersize=10, label='ST-HSC Removed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='MPP Active'),
        # Line2D([0], [0], marker='d', color='w', markerfacecolor='orangered', markersize=10, label='MPP Senescent'),
        # Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=10, label='MPP Quiescent'),
        # Line2D([0], [0], marker='s', color='w', markerfacecolor='lightcoral', markersize=10, label='MPP Removed'),
    ]
    plt.legend(handles=legend_elements, loc='best', fontsize=10)

    plt.title(f"Hierarchical Lineage Tree of Cell ID {initial_cell_id} after {time_steps} days")

plot_hierarchical_tree_with_progress(lineage, initial_cell_id)    


# Save the figure
plt.savefig(f"hierarchical_tree_cell_{initial_cell_id}.png", format="png", dpi=300)
plt.show()

