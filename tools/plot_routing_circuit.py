"""Write a checkpoint's conditional routing circuit as a standalone HTML file.

Usage:
  python tools/plot_routing_circuit.py --save-dir trained_models/v0_8_2_priorzone_CR --model-name mix_CR
  python tools/plot_routing_circuit.py --save-dir trained_models/v0_8_2_priorzone_Small --model-name mix_Small --output Plots/small_circuit.html

Only works on checkpoints trained with --prior-zone-conditioning (they carry
the per-type zone_probs table this plots). No model weights are loaded and
nothing is generated - this reads the saved config/metadata JSON only.
"""
import argparse
import magi

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--save-dir", required=True)
parser.add_argument("--model-name", required=True)
parser.add_argument("--output", default=None, help="Output HTML path; defaults next to the checkpoint.")
args = parser.parse_args()

path = magi.save_routing_circuit(args.save_dir, args.model_name, output_html=args.output)
print(f"Open with: open {path}")
