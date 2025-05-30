import json
import pandas as pd
import matplotlib.pyplot as plt

def read_log_file(path):
    """
    Reads a text file containing both single-line JSON and multiline JSON 
    objects.
    
    Args:
        path (str): Path to the file.
    
    Returns:
        list: List of parsed JSON objects.
    """
    data = []
    open_brackets_found = 0
    json_str = ""
    json_string_complete = True

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # If the current line completes the JSON object
            if json_string_complete:
                try:
                    # Attempt to parse single-line JSON
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    # Enter multiline JSON parsing
                    if "{" in line:
                        json_string_complete = False
                        open_brackets_found += line.count("{")
                    if "}" in line:
                        open_brackets_found -= line.count("}")
                    json_str += line  # Start buffering lines
            else:
                # Buffer lines for multiline JSON
                json_str += line
                if "{" in line:
                    open_brackets_found += line.count("{")
                if "}" in line:
                    open_brackets_found -= line.count("}")
                
                # Check if JSON object is complete
                if open_brackets_found == 0:
                    try:
                        data.append(json.loads(json_str))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing multiline JSON: {e}")
                    json_str = ""  # Reset buffer
                    json_string_complete = True

    return data

def split_log_file_stage1(lst):
    parameters = lst[0]
    try:
        run = parameters['run']
        model = parameters['model']
        datasets = parameters['datasets']
    except KeyError:
        run = None
        model = None
        datasets = None

    test_score = lst[-1]

    remainder = lst[1:-1]
    train_val_lst = []   
    for x in range(0, len(remainder), 2):
        train = remainder[x]
        val = remainder[x + 1]
        complete_dtc = {**train, **val}
        complete_dtc['val_agg_metrics'] = complete_dtc['val_agg_metrics'] * -1
        complete_dtc['epoch'] = x // 2
        train_val_lst.append(complete_dtc)
    return run, model, datasets, train_val_lst, test_score

def split_log_file_stage2(lst):
    parameters = lst[0]
    try:
        run = parameters['run']
        model = parameters['model']
        datasets = parameters['datasets']
    except KeyError:
        run = None
        model = None
        datasets = None

    test_score = lst[-1]

    remainder = lst[1:-1]
    train_lst = []   
    val_lst = []
    epoch=0
    for x in range(0, len(remainder)):
        dct = remainder[x]
        if 'train_loss' in dct:
            dct['epoch'] = epoch
            train_lst.append(dct)
            epoch += 1
        elif 'val_loss' in dct:
            dct['epoch'] = epoch - 1
            val_lst.append(dct)        
        else:
            return ValueError("not train and not a val, weird!")

    return run, model, datasets, train_lst, val_lst, test_score

def weighted_score_adjustment(itc_weight, itm_weight, lm_weight, 
                              train_val_lst, test_score):
    df = pd.DataFrame(train_val_lst)
    df['train_loss_itc'] = pd.to_numeric(df['train_loss_itc']) / itc_weight
    df['train_loss_itm'] = pd.to_numeric(df['train_loss_itm']) / itm_weight
    df['train_loss_lm'] = pd.to_numeric(df['train_loss_lm']) / lm_weight

    df['train_loss'] = df['train_loss_itc'] + df['train_loss_itm']\
                                                    + df['train_loss_lm']
    
    df['val_eval_loss_itc'] = pd.to_numeric(
        df['val_eval_loss_itc']
        ) / itc_weight
    df['val_eval_loss_itm'] = pd.to_numeric(
        df['val_eval_loss_itm']
        ) / itm_weight
    df['val_eval_loss_lm'] = pd.to_numeric(
        df['val_eval_loss_lm']
        ) / lm_weight

    df['val_agg_metrics'] = df['val_eval_loss_itc'] + df['val_eval_loss_itm']\
                                                    + df['val_eval_loss_lm']
    
    test_score['test_eval_loss_itc'] = pd.to_numeric(
        test_score['test_eval_loss_itc']
        ) / itc_weight
    test_score['test_eval_loss_itm'] = pd.to_numeric(
        test_score['test_eval_loss_itm']
        ) / itm_weight
    test_score['test_eval_loss_lm'] = pd.to_numeric(
        test_score['test_eval_loss_lm']
        ) / lm_weight
    test_score['test_agg_metrics'] = test_score['test_eval_loss_itc']\
                                        + test_score['test_eval_loss_itm']\
                                            + test_score['test_eval_loss_lm']

    return df, test_score

def create_2x3_plot_stage1(df):
    """
    Creates a single 2x3 figure:
    - Top row: Training and Validation stacked area plots with a unified legend
        (adjusted widths).
    - Bottom row: Individual ITC, ITM, and LM metrics with equal widths.
    
    Args:
        df (pd.DataFrame): DataFrame containing the metrics.
    """
    # Ensure numeric conversion for required columns
    numeric_cols = [
        'train_loss', 'train_loss_itc', 'train_loss_itm', 'train_loss_lm',
        'val_eval_loss_itc', 'val_eval_loss_itm', 'val_eval_loss_lm'
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Replace NaN with 0 for plotting
    df = df.fillna(0)
    
    # Create a 2x3 grid
    fig = plt.figure(figsize=(18, 12))
    # Top row with adjusted widths for stacked plots and legend
    gs_top = fig.add_gridspec(1, 3, width_ratios=[4, 1, 4], top=0.93, 
                              bottom=0.6)
    # Bottom row with equal widths for individual plots
    gs_bottom = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], top=0.53, 
                                 bottom=0.2)
    
    axs_top = [fig.add_subplot(gs_top[0, i]) for i in range(3)]
    axs_bottom = [fig.add_subplot(gs_bottom[0, i]) for i in range(3)]
    
    fig.suptitle("Loss Analysis During BLIP-2 Pretraining Stage 1", 
                 fontsize=18, fontweight='bold')
    
    axs_top[0].plot(df['epoch'], df['train_loss'], label='Total Train Loss', 
                    color='black', linewidth=2, linestyle='-')
    # Top-left: Training stacked plot
    axs_top[0].stackplot(
        df['epoch'],
        df['train_loss_itc'],
        df['train_loss_itm'],
        df['train_loss_lm'],
        labels=['Train ITC', 'Train ITM', 'Train LM'],
        colors=['lightblue', 'lightgreen', 'lightcoral'],
        alpha=0.6
    )
    axs_top[0].plot(df['epoch'], df['val_agg_metrics'], label='Total Val Loss',
                    color='darkred', linewidth=2, linestyle='--')
    axs_top[0].set_title("Training Loss Composition During Stage 1")
    axs_top[0].set_xlabel("Epoch")
    axs_top[0].set_ylabel("Loss")
    axs_top[0].grid(True)

    # Top-right: Validation stacked plot
    axs_top[2].stackplot(
        df['epoch'],
        df['val_eval_loss_itc'],
        df['val_eval_loss_itm'],
        df['val_eval_loss_lm'],
        labels=['Val ITC', 'Val ITM', 'Val LM'],
        colors=['blue', 'green', 'red'],
        alpha=0.6
    )
    axs_top[2].plot(df['epoch'], df['val_agg_metrics'], label='Total Val Loss',
                    color='darkred', linewidth=2, linestyle='--')
    axs_top[2].plot(df['epoch'], df['train_loss'], label='Total Train Loss', 
                    color='black', linewidth=2, linestyle='-')
    axs_top[2].set_title("Validation Loss Composition During Stage 1")
    axs_top[2].set_xlabel("Epoch")
    axs_top[2].set_ylabel("Loss")
    axs_top[2].grid(True)

    # Top-middle: Unified Legend
    axs_top[1].axis('off')  # Turn off axis for the legend space
    handles_0, labels_0 = axs_top[0].get_legend_handles_labels()
    handles_2, labels_2 = axs_top[2].get_legend_handles_labels()
    unique_handles_labels = dict(zip(labels_0 + labels_2, 
                                     handles_0 + handles_2))
    # Text in middle of legend
    axs_top[1].legend(unique_handles_labels.values(), 
                      unique_handles_labels.keys(), loc='center', fontsize=10, 
                      title="Loss Components")

    # Bottom-left: ITC metrics
    axs_bottom[0].plot(df['epoch'], df['train_loss_itc'], 
                       label='Train Loss (ITC)', marker='o', color='lightblue')
    axs_bottom[0].plot(df['epoch'], df['val_eval_loss_itc'], 
                       label='Val Loss (ITC)', linestyle='--', marker='x', 
                       color='blue')
    axs_bottom[0].set_title("ITC Loss Across Epochs (Training and Validation)")
    axs_bottom[0].set_xlabel("Epoch")
    axs_bottom[0].set_ylabel("Loss")
    axs_bottom[0].legend()
    axs_bottom[0].grid(True)

    # Bottom-middle: ITM metrics
    axs_bottom[1].plot(df['epoch'], df['train_loss_itm'], 
                       label='Train Loss (ITM)', marker='o', 
                       color='lightgreen')
    axs_bottom[1].plot(df['epoch'], df['val_eval_loss_itm'], 
                       label='Val Loss (ITM)', linestyle='--', marker='x', 
                       color='green')
    axs_bottom[1].set_title("ITM Loss Across Epochs (Training and Validation)")
    axs_bottom[1].set_xlabel("Epoch")
    axs_bottom[1].set_ylabel("Loss")
    axs_bottom[1].legend()
    axs_bottom[1].grid(True)

    # Bottom-right: LM metrics
    axs_bottom[2].plot(df['epoch'], df['train_loss_lm'], 
                       label='Train Loss (LM)', marker='o', color='lightcoral')
    axs_bottom[2].plot(df['epoch'], df['val_eval_loss_lm'], 
                       label='Val Loss (LM)', linestyle='--', marker='x', 
                       color='red')
    axs_bottom[2].set_title("LM Loss Across Epochs (Training and Validation)")
    axs_bottom[2].set_xlabel("Epoch")
    axs_bottom[2].set_ylabel("Loss")
    axs_bottom[2].legend()
    axs_bottom[2].grid(True)

    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    plt.show()

# def plot_training_and_validation_stage2(train_lst, val_lst):
#     """
#     Visualize training and validation metrics during Stage 2 with consistent 
#     colors, markers, and styles
#     while using full metric names and logical color gradation for BLEU scores.
    
#     Parameters:
#     - train_lst: List of dictionaries containing training data.
#     - val_lst: List of dictionaries containing validation data.
    
#     Outputs:
#     - Three plots: Training/Validation loss, Stacked Aggregated Metrics, and 
#         Detailed Metrics.
#     """
#     # Convert to DataFrame
#     train_df = pd.DataFrame(train_lst).astype({'train_loss': float, 
#                                                'epoch': int})
#     val_df = pd.DataFrame(val_lst).astype({
#         'val_loss': float, 'val_BLEU_1': float, 'val_BLEU_2': float, 
#         'val_BLEU_3': float, 'val_BLEU_4': float, 'val_METEOR': float, 
#         'val_ROUGE_L': float, 'val_CIDEr': float, 'epoch': int
#     })
    
#     # # Compute aggregated metrics (sum of individual metrics)
#     # val_df['aggregated_metrics'] = (
#     #     val_df['val_BLEU_1'] + val_df['val_BLEU_2'] + val_df['val_BLEU_3'] +
#     #     val_df['val_BLEU_4'] + val_df['val_METEOR'] + val_df['val_ROUGE_L'] + 
#     #     val_df['val_CIDEr']
#     # )
    
#     # Compute aggregated metrics (sum of individual metrics)
#     val_df['aggregated_metrics'] = (
#         val_df['val_BLEU_1'] +
#         val_df['val_BLEU_4'] + val_df['val_METEOR'] + val_df['val_ROUGE_L'] + 
#         val_df['val_CIDEr']
#     )
    

#     # Set up the figure
#     fig, axs = plt.subplots(3, 1, figsize=(7, 13))
    
#     # Plot 1: Training and Validation Loss
#     axs[0].plot(train_df['epoch'], train_df['train_loss'], label='Train Loss', 
#                 marker='', color='black', linewidth=2, linestyle='-')
#     axs[0].plot(val_df['epoch'], val_df['val_loss'], label='Validation Loss', 
#                 marker='x', color='darkred', linewidth=2, linestyle='--')
#     axs[0].set_title('Training and Validation Loss', fontsize=14)
#     axs[0].set_xlabel('Epoch', fontsize=12)
#     axs[0].set_ylabel('Loss', fontsize=12)
#     axs[0].legend(fontsize=10)
#     axs[0].grid()

#     # # Plot 2: Aggregated Metrics
#     # axs[1].stackplot(
#     #     val_df['epoch'], 
#     #     val_df['val_BLEU_1'], val_df['val_BLEU_2'], val_df['val_BLEU_3'], 
#     #     val_df['val_BLEU_4'], val_df['val_METEOR'], val_df['val_ROUGE_L'], 
#     #     val_df['val_CIDEr'], 
#     #     labels=['BLEU@1', 'BLEU@2', 'BLEU@3', 'BLEU@4', 'METEOR', 'ROUGE-L', 
#     #             'CIDEr'],
#     #     colors=['lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 
#     #             'lightcoral', 'gray', 'gold'],
#     #     alpha=0.6
#     # )

#     # Plot 2: Aggregated Metrics
#     axs[1].stackplot(
#         val_df['epoch'], 
#         val_df['val_BLEU_1'], 
#         val_df['val_BLEU_4'], val_df['val_METEOR'], val_df['val_ROUGE_L'], 
#         val_df['val_CIDEr'], 
#         labels=['BLEU@1', 'BLEU@4', 'METEOR', 'ROUGE-L', 
#                 'CIDEr'],
#         colors=['lightblue', 'dodgerblue', 
#                 'lightcoral', 'gray', 'gold'],
#         alpha=0.6
#     )

#     axs[1].plot(val_df['epoch'], val_df['aggregated_metrics'], 
#                 label='Total', marker='s', color='green', 
#                 linewidth=2, linestyle='-')
#     axs[1].set_title('Captioning Metrics (Stacked)', 
#                      fontsize=14)
#     axs[1].set_xlabel('Epoch', fontsize=12)
#     axs[1].set_ylabel('Metric Value', fontsize=12)
#     axs[1].legend(loc='upper left', fontsize=10)
#     axs[1].grid()

#     # # Plot 3: Detailed Metrics Line Plot
#     # for metric, color, label in zip(
#     #     ['val_BLEU_1', 'val_BLEU_2', 'val_BLEU_3', 'val_BLEU_4', 'val_METEOR', 
#     #      'val_ROUGE_L', 'val_CIDEr'],
#     #     ['lightblue', 'skyblue', 'deepskyblue', 'dodgerblue', 'lightcoral', 
#     #      'gray', 'gold'],
#     #     ['BLEU@1', 'BLEU@2', 'BLEU@3', 'BLEU@4', 'METEOR', 'ROUGE-L', 'CIDEr']
#     # ):
#     # Plot 3: Detailed Metrics Line Plot
#     for metric, color, label in zip(
#         ['val_BLEU_1', 'val_BLEU_4', 'val_METEOR', 
#          'val_ROUGE_L', 'val_CIDEr'],
#         ['lightblue', 'dodgerblue', 'lightcoral', 
#          'gray', 'gold'],
#         ['BLEU@1', 'BLEU@4', 'METEOR', 'ROUGE-L', 'CIDEr']
#     ):
#         axs[2].plot(val_df['epoch'], val_df[metric], label=label, marker='o', 
#                     linestyle='-', color=color, linewidth=2)
#     axs[2].set_title('Captioning Metrics (Individually)', fontsize=14)
#     axs[2].set_xlabel('Epoch', fontsize=12)
#     axs[2].set_ylabel('Metric Value', fontsize=12)
#     axs[2].legend(fontsize=10)
#     axs[2].grid()

#     axs[0].set_ylim((1.0, 2.6))
#     axs[1].set_ylim((0, 1.4))
#     axs[2].set_ylim((0, 0.4))

#     axs[0].legend(loc='upper right', fontsize=8)
#     axs[1].legend(loc='upper left', fontsize=8)
#     axs[2].legend(loc='lower right', fontsize=8)


#     # Adjust layout and show plots
#     plt.tight_layout()
#     plt.show()

def plot_training_and_validation_stage2(train_lst, val_lst):
    """
    Visualize training and validation metrics during Stage 2 with a layout
    of one plot at the top and two plots at the bottom.
    
    Parameters:
    - train_lst: List of dictionaries containing training data.
    - val_lst: List of dictionaries containing validation data.
    
    Outputs:
    - Top Plot: Training and Validation Loss.
    - Bottom Left: Stacked Captioning Metrics.
    - Bottom Right: Detailed Captioning Metrics.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Convert lists to DataFrames and set appropriate types
    train_df = pd.DataFrame(train_lst).astype({'train_loss': float, 
                                                 'epoch': int})
    val_df = pd.DataFrame(val_lst).astype({
        'val_loss': float, 
        'val_BLEU_1': float, 
        'val_BLEU_4': float, 
        'val_METEOR': float, 
        'val_ROUGE_L': float, 
        'val_CIDEr': float, 
        'epoch': int
    })
    
    # Compute aggregated metrics (sum of individual metrics)
    val_df['aggregated_metrics'] = (
        val_df['val_BLEU_1'] +
        val_df['val_BLEU_4'] + 
        val_df['val_METEOR'] + 
        val_df['val_ROUGE_L'] + 
        val_df['val_CIDEr']
    )
    
    # Create figure with GridSpec: 2 rows, 2 columns.
    # The top plot will span both columns.
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    
    # Top axis: spans both columns
    ax_top = fig.add_subplot(gs[0, :])
    # Bottom left and right axes
    ax_bottom_left = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])
    
    # Top Plot: Training and Validation Loss
    ax_top.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss',
                marker='', color='black', linewidth=2, linestyle='-')
    ax_top.plot(val_df['epoch'], val_df['val_loss'], label='Validation Loss',
                marker='x', color='darkred', linewidth=2, linestyle='--')
    ax_top.set_title('Training and Validation Loss', fontsize=14)
    ax_top.set_xlabel('Epoch', fontsize=12)
    ax_top.set_ylabel('Loss', fontsize=12)
    ax_top.legend(loc='upper right', fontsize=10)
    ax_top.grid()
    ax_top.set_ylim((1.0, 2.6))
    
    # Bottom Left Plot: Stacked Captioning Metrics
    ax_bottom_left.stackplot(
        val_df['epoch'], 
        val_df['val_BLEU_1'], 
        val_df['val_BLEU_4'], 
        val_df['val_METEOR'], 
        val_df['val_ROUGE_L'], 
        val_df['val_CIDEr'], 
        labels=['BLEU@1', 'BLEU@4', 'METEOR', 'ROUGE-L', 'CIDEr'],
        # Colors here are the defaults; you can remove the colors if preferred.
        colors=['lightblue', 'dodgerblue', 'lightcoral', 'gray', 'gold'],
        alpha=0.6
    )
    ax_bottom_left.plot(val_df['epoch'], val_df['aggregated_metrics'], 
                        label='Total', marker='s', color='green', 
                        linewidth=2, linestyle='-')
    ax_bottom_left.set_title('Captioning Metrics (Stacked)', fontsize=14)
    ax_bottom_left.set_xlabel('Epoch', fontsize=12)
    ax_bottom_left.set_ylabel('Metric Value', fontsize=12)
    ax_bottom_left.legend(loc='upper left', fontsize=10)
    ax_bottom_left.grid()
    ax_bottom_left.set_ylim((0, 1.4))
    
    # Bottom Right Plot: Detailed Captioning Metrics (Line Plot)
    for metric, color, label in zip(
        ['val_BLEU_1', 'val_BLEU_4', 'val_METEOR', 'val_ROUGE_L', 'val_CIDEr'],
        ['lightblue', 'dodgerblue', 'lightcoral', 'gray', 'gold'],
        ['BLEU@1', 'BLEU@4', 'METEOR', 'ROUGE-L', 'CIDEr']
    ):
        ax_bottom_right.plot(val_df['epoch'], val_df[metric], label=label, 
                             marker='o', linestyle='-', color=color, linewidth=2)
    ax_bottom_right.set_title('Captioning Metrics (Individually)', fontsize=14)
    ax_bottom_right.set_xlabel('Epoch', fontsize=12)
    ax_bottom_right.set_ylabel('Metric Value', fontsize=12)
    ax_bottom_right.legend(loc='lower right', fontsize=10)
    ax_bottom_right.grid()
    ax_bottom_right.set_ylim((0, 0.4))
    
    # Adjust layout and show the plots
    plt.tight_layout()
    plt.show()
