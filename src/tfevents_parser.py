import os
import pandas as pd
from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
from pathlib import Path
import json

def parse_tfevents_file(file_path):
    """
    Parse a single tfevents file and extract scalar values
    
    Args:
        file_path: Path to the .tfevents file
        
    Returns:
        List of dictionaries containing the extracted data
    """
    data = []
    
    try:
        for event in summary_iterator(str(file_path)):
            if event.HasField('summary'):
                for value in event.summary.value:
                    if value.HasField('simple_value'):  # Scalar values
                        data.append({
                            'timestamp': event.wall_time,
                            'step': event.step,
                            'tag': value.tag,
                            'value': value.simple_value
                        })
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        
    return data

def parse_tfevents_directory(directory_path, output_format='csv'):
    """
    Parse all tfevents files in a directory
    
    Args:
        directory_path: Path to directory containing tfevents files
        output_format: 'csv', 'json', or 'pickle'
        
    Returns:
        DataFrame containing all extracted data
    """
    directory_path = Path(directory_path)
    all_data = []
    
    # Find all tfevents files recursively
    tfevents_files = list(directory_path.rglob('*.tfevents.*'))
    
    if not tfevents_files:
        print(f"No tfevents files found in {directory_path}")
        return pd.DataFrame()
    
    print(f"Found {len(tfevents_files)} tfevents files")
    
    for file_path in tfevents_files:
        print(f"Parsing {file_path.name}...")
        file_data = parse_tfevents_file(file_path)
        
        # Add source file info including relative path and experiment ID
        for entry in file_data:
            entry['source_file'] = file_path.name
            entry['relative_path'] = str(file_path.relative_to(directory_path))
            # Extract experiment name from path (e.g., 'exp3' from 'exp3/logs/events...')
            path_parts = file_path.relative_to(directory_path).parts
            entry['experiment'] = path_parts[0] if path_parts else 'unknown'
            
        all_data.extend(file_data)
    
    if not all_data:
        print("No data extracted from tfevents files")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Convert timestamp to datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Sort by step and timestamp
    df = df.sort_values(['step', 'timestamp']).reset_index(drop=True)
    
    print(f"Extracted {len(df)} data points")
    print(f"Experiments: {df['experiment'].unique().tolist()}")
    print(f"Unique metrics: {df['tag'].unique().tolist()}")
    print(f"Step range: {df['step'].min()} - {df['step'].max()}")
    
    return df

def save_data(df, output_path, format_type='csv'):
    """Save the extracted data in the specified format"""
    output_path = Path(output_path)
    
    if format_type == 'csv':
        df.to_csv(output_path.with_suffix('.csv'), index=False)
        print(f"Data saved to {output_path.with_suffix('.csv')}")
    elif format_type == 'json':
        df.to_json(output_path.with_suffix('.json'), orient='records', indent=2)
        print(f"Data saved to {output_path.with_suffix('.json')}")
    elif format_type == 'pickle':
        df.to_pickle(output_path.with_suffix('.pkl'))
        print(f"Data saved to {output_path.with_suffix('.pkl')}")
    else:
        raise ValueError("Format must be 'csv', 'json', or 'pickle'")

def generate_summary_stats(df):
    """Generate summary statistics for the extracted data"""
    if df.empty:
        return "No data to summarize"
    
    summary = {
        'total_data_points': len(df),
        'unique_metrics': df['tag'].nunique(),
        'metrics_list': df['tag'].unique().tolist(),
        'step_range': {
            'min': int(df['step'].min()),
            'max': int(df['step'].max()),
            'count': df['step'].nunique()
        },
        'time_range': {
            'start': df['datetime'].min().isoformat(),
            'end': df['datetime'].max().isoformat(),
            'duration_hours': (df['datetime'].max() - df['datetime'].min()).total_seconds() / 3600
        }
    }
    
    # Per-metric statistics
    metric_stats = {}
    for metric in df['tag'].unique():
        metric_data = df[df['tag'] == metric]['value']
        metric_stats[metric] = {
            'count': len(metric_data),
            'mean': float(metric_data.mean()),
            'std': float(metric_data.std()),
            'min': float(metric_data.min()),
            'max': float(metric_data.max())
        }
    
    summary['metric_statistics'] = metric_stats
    
    return summary

def main():
    parser = argparse.ArgumentParser(description='Extract data from TensorFlow events files')
    parser.add_argument('input_path', help='Path to tfevents file or directory containing tfevents files')
    parser.add_argument('-o', '--output', default=None, help='Output file path (without extension). If not specified, saves to input directory as "extracted_data"')
    parser.add_argument('-f', '--format', choices=['csv', 'json', 'pickle'], default='csv', help='Output format')
    parser.add_argument('--summary', action='store_true', help='Generate summary statistics')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    # Set default output path to be in the input directory
    if args.output is None:
        if input_path.is_file():
            args.output = input_path.parent / "data"
        else:
            args.output = input_path / "data"
    else:
        args.output = Path(args.output)
    
    if input_path.is_file():
        # Single file
        print(f"Parsing single file: {input_path}")
        data = parse_tfevents_file(input_path)
        df = pd.DataFrame(data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            df['source_file'] = input_path.name
    elif input_path.is_dir():
        # Directory
        print(f"Parsing directory: {input_path}")
        df = parse_tfevents_directory(input_path, args.format)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return
    
    if df.empty:
        print("No data extracted")
        return
    
    # Save data
    save_data(df, args.output, args.format)
    
    # Generate summary if requested
    if args.summary:
        summary = generate_summary_stats(df)
        summary_path = Path(args.output).with_suffix('.summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to {summary_path}")
        
        # Print brief summary to console
        print("\n--- Summary ---")
        print(f"Total data points: {summary['total_data_points']}")
        print(f"Unique metrics: {summary['unique_metrics']}")
        print(f"Metrics: {', '.join(summary['metrics_list'])}")
        print(f"Steps: {summary['step_range']['min']} - {summary['step_range']['max']}")

if __name__ == "__main__":
    main()