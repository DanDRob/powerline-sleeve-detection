#!/usr/bin/env python3
import os
import ast
import sys
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import json
import plotly.graph_objects as go
import networkx as nx

# Set a flag for visualization availability
VISUALIZATION_AVAILABLE = True


class CodebaseAnalyzer:
    """Analyzes Python codebase structure and function relationships."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.function_calls = {}  # Maps function callers to called functions
        self.function_defs = {}   # Maps modules to defined functions
        self.imports = {}         # Maps modules to their imports
        self.module_hierarchy = {}  # Maps modules to their parent modules

    def analyze(self):
        """Analyze the entire codebase."""
        self._scan_directory_structure()
        self._analyze_python_files()

    def _scan_directory_structure(self):
        """Scan and store the directory structure."""
        self.dir_structure = {}

        def scan_dir(dir_path, structure):
            dir_items = {}

            for item in sorted(os.listdir(dir_path)):
                item_path = os.path.join(dir_path, item)

                # Skip hidden directories, __pycache__, __init__ files, and build.lib directories
                if (item.startswith('.') or
                    item == '__pycache__' or
                    item == '__init__.py' or
                        'build.lib' in item_path):
                    continue

                if os.path.isdir(item_path):
                    # Skip build.lib directories
                    if 'build.lib' in item_path:
                        continue

                    sub_items = {}
                    scan_dir(item_path, sub_items)
                    dir_items[item + '/'] = sub_items
                else:
                    if item.endswith('.py') and item != '__init__.py':
                        # For Python files, also store their path for later analysis
                        rel_path = os.path.relpath(item_path, self.root_dir)
                        self.module_hierarchy[rel_path] = item
                    dir_items[item] = None

            structure.update(dir_items)

        scan_dir(self.root_dir, self.dir_structure)

    def _analyze_python_files(self):
        """Analyze Python files for functions and dependencies."""
        for root, _, files in os.walk(self.root_dir):
            # Skip build.lib directories
            if 'build.lib' in root:
                continue

            for file in files:
                # Skip __init__.py files
                if file.endswith('.py') and file != '__init__.py':
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, self.root_dir)
                    self._analyze_file(file_path, rel_path)

    def _analyze_file(self, file_path: str, rel_path: str):
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = ast.parse(content)

            # Module name calculation (converting path to module notation)
            module_name = rel_path.replace(os.sep, '.').replace('.py', '')

            # Extract function definitions
            functions = []
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)

            self.function_defs[module_name] = functions

            # Extract imports
            module_imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        module_imports.append(name.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_imports.append(node.module)

            self.imports[module_name] = module_imports

            # Analyze function calls
            function_calls = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        function_calls.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        if isinstance(node.func.value, ast.Name):
                            # This handles calls like module.function()
                            function_calls.append(
                                f"{node.func.value.id}.{node.func.attr}")

            self.function_calls[module_name] = function_calls

        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def generate_directory_tree(self) -> str:
        """Generate a text representation of the directory tree."""
        def print_tree(structure, prefix="", last=True):
            output = []
            items = list(structure.items())

            for i, (name, sub_items) in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "

                line = f"{prefix}{connector}{name}"
                output.append(line)

                if sub_items is not None:
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    sub_output = print_tree(sub_items, new_prefix)
                    output.extend(sub_output)

            return output

        tree_lines = print_tree(self.dir_structure)
        return "\n".join(["Project Directory Structure:", ""] + tree_lines)

    def generate_function_flow(self) -> str:
        """Generate a text representation of function flow."""
        output = ["Function Flow Analysis:", ""]

        # First, list all modules and their defined functions
        output.append("Modules and Defined Functions:")
        for module, functions in sorted(self.function_defs.items()):
            output.append(f"  {module}:")
            for func in sorted(functions):
                output.append(f"    - {func}")

        output.append("")

        # Then, show imports between modules
        output.append("Module Dependencies:")
        for module, imports in sorted(self.imports.items()):
            if imports:
                output.append(f"  {module} imports:")
                for imp in sorted(imports):
                    output.append(f"    - {imp}")

        output.append("")

        # Finally, show function calls
        output.append("Function Calls:")
        for module, calls in sorted(self.function_calls.items()):
            if calls:
                output.append(f"  In {module}:")
                # Group and count repeated calls
                call_counts = {}
                for call in calls:
                    call_counts[call] = call_counts.get(call, 0) + 1

                for call, count in sorted(call_counts.items()):
                    output.append(f"    - {call} (called {count} times)")

        return "\n".join(output)

    def generate_visualization(self, output_file: str = "codebase_flow.html"):
        """Generate a visual representation of module dependencies."""
        try:
            G = nx.DiGraph()

            excluded_keywords = ['build.lib', 'logging', 'config']

            for module in self.function_defs:
                if not any(keyword in module for keyword in excluded_keywords):
                    G.add_node(module)

            for module, imports in self.imports.items():
                if any(keyword in module for keyword in excluded_keywords):
                    continue
                for imp in imports:
                    for potential_match in self.function_defs.keys():
                        if potential_match.endswith(imp) or potential_match == imp:
                            if not any(keyword in potential_match for keyword in excluded_keywords):
                                G.add_edge(module, potential_match)

            pos = nx.spring_layout(G, seed=42)

            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines')

            node_x, node_y, text = [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                text.append(node)

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=text,
                textposition="bottom center",
                marker=dict(
                    showscale=True,
                    colorscale='YlGnBu',
                    size=10,
                    color=[],
                    line_width=2))

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='Module Dependencies',
                                title_x=0.5,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                annotations=[dict(
                                    text="Module dependency graph",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002)],
                                xaxis=dict(
                                    showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

            fig.write_html(output_file)
            print(f"Visualization saved to {output_file}")
            return output_file

        except ImportError as e:
            print(f"Plotly or NetworkX not installed: {e}")
            return None
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return None

    def generate_ascii_graph(self) -> str:
        """Generate a simple ASCII art representation of the module dependencies graph."""
        output = ["ASCII Module Dependency Graph:", ""]

        # Build adjacency list of imports
        adjacency = {}
        for module in sorted(self.function_defs.keys()):
            adjacency[module] = []

        for module, imports in self.imports.items():
            for imp in imports:
                for potential_match in self.function_defs.keys():
                    if potential_match.endswith(imp) or potential_match == imp:
                        if module in adjacency:
                            adjacency[module].append(potential_match)

        # Simplify module names for display
        simple_names = {}
        for i, module in enumerate(sorted(adjacency.keys())):
            # Get the last part of the module path
            simple_name = module.split('.')[-1] if '.' in module else module
            # Add a unique index to avoid duplicates
            identifier = f"{simple_name}_{i}"
            simple_names[module] = identifier

        # Print modules and their connections
        for module, connections in sorted(adjacency.items()):
            if connections:
                output.append(
                    f"{simple_names[module]} --> {', '.join(simple_names[conn] for conn in connections if conn in simple_names)}")
            else:
                output.append(f"{simple_names[module]} (no dependencies)")

        return "\n".join(output)

    def generate_mermaid_diagram(self) -> str:
        """Generate a Mermaid.js flowchart diagram of module dependencies."""
        output = ["```mermaid", "flowchart TD"]

        # Add nodes for each module
        for i, module in enumerate(sorted(self.function_defs.keys())):
            node_id = f"mod{i}"
            # Simplify module name for display
            display_name = module.split('.')[-1] if '.' in module else module
            output.append(f"    {node_id}[\"{display_name}\"]")

        output.append("")

        # Add edges for imports
        edge_count = 0
        module_to_id = {module: f"mod{i}" for i,
                        module in enumerate(sorted(self.function_defs.keys()))}

        for module, imports in sorted(self.imports.items()):
            if module not in module_to_id:
                continue

            for imp in imports:
                # Find if this import matches any of our modules
                for potential_match in sorted(self.function_defs.keys()):
                    if potential_match.endswith(imp) or potential_match == imp:
                        if potential_match in module_to_id:
                            source = module_to_id[module]
                            target = module_to_id[potential_match]
                            output.append(f"    {source} --> {target}")
                            edge_count += 1

                            # Avoid generating too complex diagrams
                            if edge_count >= 100:
                                output.append(
                                    "    %% Note: Some edges omitted for clarity")
                                break

                if edge_count >= 100:
                    break

            if edge_count >= 100:
                break

        output.append("```")
        return "\n".join(output)

    def extract_core_workflow(self) -> str:
        """Extract and visualize the core business logic workflow."""
        # Identify key modules
        core_modules = [
            m for m in self.function_defs.keys()
            if any(segment in m for segment in [
                "cli", "detector", "tracker", "model", "processor",
                "route_planner", "train_model", "evaluate_model"
            ])
        ]

        output = ["## Core Business Logic Flows", ""]

        # Extract main entry points
        entry_points = {}
        for module in core_modules:
            if "cli.py" in module:
                # CLI commands are entry points
                commands = [f for f in self.function_defs.get(module, [])
                            if f.endswith("_command")]
                if commands:
                    entry_points["Command Line Interface"] = {
                        "module": module,
                        "functions": commands
                    }
            elif "train_model.py" in module:
                # Training functions
                entry_points["Model Training"] = {
                    "module": module,
                    "functions": [f for f in self.function_defs.get(module, [])
                                  if any(keyword in f for keyword in
                                         ["train", "prepare", "augment", "tune"])]
                }
            elif "evaluate_model.py" in module:
                # Evaluation functions
                entry_points["Model Evaluation"] = {
                    "module": module,
                    "functions": [f for f in self.function_defs.get(module, [])
                                  if any(keyword in f for keyword in
                                         ["evaluate", "compare"])]
                }
            elif "processor" in module:
                # Processing functions
                entry_points["Batch Processing"] = {
                    "module": module,
                    "functions": [f for f in self.function_defs.get(module, [])
                                  if "process" in f]
                }
            elif "detector" in module:
                # Detection functions
                entry_points["Detection"] = {
                    "module": module,
                    "functions": [f for f in self.function_defs.get(module, [])
                                  if "detect" in f]
                }

        # Generate flow diagrams for each workflow
        output.append("### Main Application Workflows\n")

        for workflow_name, workflow_info in entry_points.items():
            output.append(f"#### {workflow_name}\n")
            output.append(f"Module: `{workflow_info['module']}`\n")
            output.append("Entry Points:")
            for func in workflow_info['functions']:
                output.append(f"- `{func}()`")
            output.append("")

            # Add workflow diagram in Mermaid.js format
            if workflow_name == "Command Line Interface":
                output.append(self._generate_cli_workflow())
            elif workflow_name == "Model Training":
                output.append(self._generate_training_workflow())
            elif workflow_name == "Model Evaluation":
                output.append(self._generate_evaluation_workflow())
            elif workflow_name == "Batch Processing":
                output.append(self._generate_processing_workflow())
            elif workflow_name == "Detection":
                output.append(self._generate_detection_workflow())

            output.append("")

        return "\n".join(output)

    def _generate_cli_workflow(self) -> str:
        """Generate a workflow diagram for the CLI interface."""
        diagram = [
            "```mermaid",
            "flowchart TD",
            "    main[main()] --> parseArgs[parse_args()]",
            "    parseArgs --> cmdCheck{Command type?}",
            "    cmdCheck -->|process| processCmd[process_command()]",
            "    cmdCheck -->|plan| planCmd[plan_command()]",
            "    cmdCheck -->|train| trainCmd[train_command()]",
            "",
            "    processCmd --> initProcessor[Initialize BatchProcessor]",
            "    processCmd --> procRoute[process_route/process_batch]",
            "",
            "    planCmd --> initPlanner[Initialize RoutePlanner]",
            "    planCmd --> planRoute[plan_route/plan_routes_from_csv]",
            "",
            "    trainCmd --> buildCmd[Build training command]",
            "    trainCmd --> execTrain[Execute training script]",
            "```"
        ]
        return "\n".join(diagram)

    def _generate_training_workflow(self) -> str:
        """Generate a workflow diagram for model training."""
        diagram = [
            "```mermaid",
            "flowchart TD",
            "    main[main()] --> parseArgs[parse_args()]",
            "    parseArgs --> loadConfig[load_config()]",
            "    loadConfig --> cmdCheck{Command?}",
            "",
            "    cmdCheck -->|train| trainCmd[run_train_command()]",
            "    trainCmd --> trainMode{Mode?}",
            "",
            "    trainMode -->|prepare| prepData[prepare_dataset()]",
            "    trainMode -->|augment| augData[augment_dataset()]",
            "    trainMode -->|train| trainModel[train_model()]",
            "    trainMode -->|tune| tuneHyper[tune_hyperparameters()]",
            "    trainMode -->|auto-label| autoLabel[auto_label_images()]",
            "    trainMode -->|create-ensemble| createEnsemble[create_ensemble()]",
            "    trainMode -->|full-workflow| fullWorkflow[run_full_workflow()]",
            "",
            "    prepData --> dataMgr[DataManager operations]",
            "    augData --> augPipeline[AugmentationPipeline operations]",
            "    trainModel --> trainer[SleeveModelTrainer operations]",
            "    tuneHyper --> hyperTuner[HyperparameterTuner operations]",
            "    autoLabel --> labeler[AutoLabeler operations]",
            "    createEnsemble --> ensembleDetector[EnsembleDetector operations]",
            "```"
        ]
        return "\n".join(diagram)

    def _generate_evaluation_workflow(self) -> str:
        """Generate a workflow diagram for model evaluation."""
        diagram = [
            "```mermaid",
            "flowchart TD",
            "    main[main()] --> parseArgs[parse_args()]",
            "    parseArgs --> loadConfig[load_config()]",
            "    loadConfig --> cmdCheck{Command?}",
            "",
            "    cmdCheck -->|evaluate| evalCmd[run_evaluate_command()]",
            "    evalCmd --> evalMode{Mode?}",
            "",
            "    evalMode -->|single| evalSingle[evaluate_single_model()]",
            "    evalMode -->|ensemble| evalEnsemble[evaluate_ensemble()]",
            "    evalMode -->|compare| compareModels[compare_models()]",
            "",
            "    evalSingle --> evalType{Evaluation type?}",
            "    evalType -->|YOLO val| evalYolo[evaluate_with_yolo_val()]",
            "    evalType -->|Test dir| evalTest[evaluate_on_test_dir()]",
            "",
            "    evalYolo --> metrics[Calculate metrics]",
            "    evalTest --> loadLabels[load_yolo_labels()]",
            "    loadLabels --> matchPred[match_predictions_to_ground_truth()]",
            "    matchPred --> calcMetric[calculate_metrics()]",
            "    calcMetric --> visualization[create_metric_plots()]",
            "```"
        ]
        return "\n".join(diagram)

    def _generate_processing_workflow(self) -> str:
        """Generate a workflow diagram for batch processing."""
        diagram = [
            "```mermaid",
            "flowchart TD",
            "    procBatch[process_batch_from_csv()] --> loadRoutes[load_routes_from_csv()]",
            "    loadRoutes --> routeCheck{Process in parallel?}",
            "    routeCheck -->|Yes| procParallel[process_routes_parallel()]",
            "    routeCheck -->|No| procSequential[process_routes_sequential()]",
            "",
            "    procParallel --> processPool[Create processing pool]",
            "    processPool --> execRoutes[Execute route processing tasks]",
            "",
            "    procSequential --> loopRoutes[Loop through routes]",
            "    loopRoutes --> procSingleRoute[process_route()]",
            "",
            "    procSingleRoute --> planRoute[Plan route]",
            "    planRoute --> detectSleeves[Detect sleeves]",
            "    detectSleeves --> trackObjects[Track objects]",
            "    trackObjects --> saveResults[Save results]",
            "    saveResults --> generateReport[Generate report]",
            "```"
        ]
        return "\n".join(diagram)

    def _generate_detection_workflow(self) -> str:
        """Generate a workflow diagram for detection."""
        diagram = [
            "```mermaid",
            "flowchart TD",
            "    detect[detect()] --> loadModel[Load model if needed]",
            "    loadModel --> modelType{Model type?}",
            "    modelType -->|YOLOv8| detectYolo8[_detect_yolov8()]",
            "    modelType -->|YOLOv5| detectYolo5[_detect_yolov5()]",
            "    modelType -->|EfficientDet| detectEffDet[_detect_efficientdet()]",
            "",
            "    detectYolo8 --> processResults[_process_results()]",
            "    detectYolo5 --> processResults",
            "    detectEffDet --> processResults",
            "",
            "    batchDetect[batch_detect()] --> useEnsemble{Use ensemble?}",
            "    useEnsemble -->|Yes| detectEnsemble[detect_ensemble()]",
            "    useEnsemble -->|No| detectLoop[Loop through images]",
            "    detectLoop --> detect",
            "",
            "    detectEnsemble --> modelIntegration[EnsembleIntegration]",
            "    modelIntegration --> weightedBoxes[weighted_boxes_fusion()]",
            "```"
        ]
        return "\n".join(diagram)

    def generate_report(self, output_file: str = "codebase_analysis.md"):
        """Generate a comprehensive report of the codebase analysis."""
        dir_tree = self.generate_directory_tree()
        function_flow = self.generate_function_flow()
        mermaid_diagram = self.generate_mermaid_diagram()
        core_workflow = self.extract_core_workflow()
        ascii_graph = self.generate_ascii_graph()

        # Generate the visualization if libraries are available
        vis_file = self.generate_visualization() if VISUALIZATION_AVAILABLE else None

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# Powerline Sleeve Detection Codebase Analysis\n\n")

            f.write("> Note: `__init__.py` files and `build.lib` directories have been excluded from this analysis to focus on core functionality.\n\n")

            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(dir_tree)
            f.write("\n```\n\n")

            f.write(core_workflow)
            f.write("\n\n")

            f.write("## Module Dependencies\n\n")
            f.write(
                "The following diagram shows the dependencies between modules:\n\n")
            f.write(mermaid_diagram)
            f.write("\n\n")

            if vis_file:
                f.write(
                    f"![Module Dependencies Visualization]({os.path.basename(vis_file)})\n\n")
            else:
                f.write(
                    "*Matplotlib visualization could not be generated due to NumPy compatibility issues. Using text-based visualization instead:*\n\n")
                f.write("```\n")
                f.write(ascii_graph)
                f.write("\n```\n\n")

            f.write("## Function Analysis\n\n")
            f.write("```\n")
            f.write(function_flow)
            f.write("\n```\n\n")

            f.write("## Summary\n\n")
            f.write(f"Total modules: {len(self.function_defs)}\n")

            total_functions = sum(len(funcs)
                                  for funcs in self.function_defs.values())
            f.write(f"Total functions: {total_functions}\n")

            total_imports = sum(len(imps) for imps in self.imports.values())
            f.write(f"Total imports: {total_imports}\n")

        return output_file


def main():
    root_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    analyzer = CodebaseAnalyzer(root_dir)
    analyzer.analyze()
    report_file = analyzer.generate_report()
    print(f"Analysis complete. Report saved to {report_file}")


if __name__ == "__main__":
    main()
