"""AFNI integration for fmrimod.

This module provides functions to convert contrasts to AFNI GLT format
and write them to files for use with AFNI's 3dDeconvolve and 3dREMLfit.
"""

from __future__ import annotations

from functools import singledispatch
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np


@singledispatch
def to_glt(x, **kwargs) -> Dict[str, Any]:
    """Convert a contrast to AFNI GLT format.
    
    This is a generic function that converts various contrast objects
    to AFNI GLT (General Linear Test) format strings.
    
    Parameters
    ----------
    x : object
        Contrast object to convert
    **kwargs
        Additional arguments for specific methods
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'glt_str': GLT string(s) in AFNI format
        - 'name': GLT name(s) for AFNI commands
        - 'con': Original contrast object
    
    Examples
    --------
    >>> from fmrimod import contrast
    >>> con = contrast("condition[A] - condition[B]", name="A_vs_B")
    >>> glt = to_glt(con)  # doctest: +SKIP
    >>> print(glt['glt_str'])  # doctest: +SKIP
    '1*condition_A -1*condition_B'
    """
    raise NotImplementedError(f"to_glt not implemented for {type(x)}")


def write_glt(glt: Dict[str, Any], fname: Optional[Union[str, Path]] = None) -> None:
    """Write GLT contrast to file.
    
    Parameters
    ----------
    glt : dict
        GLT dictionary from to_glt() containing:
        - 'glt_str': GLT string(s) in AFNI format
        - 'name': GLT name(s) 
        - 'con': Original contrast object
    fname : str or Path, optional
        Output filename. If None, uses the GLT name + '.txt'
    
    Examples
    --------
    >>> from fmrimod import contrast, to_glt, write_glt
    >>> con = contrast("condition[A] - condition[B]", name="A_vs_B")
    >>> glt = to_glt(con)  # doctest: +SKIP
    >>> write_glt(glt, "my_contrast.txt")  # doctest: +SKIP
    """
    if 'glt_str' not in glt:
        raise ValueError("GLT dictionary must contain 'glt_str' key")
    
    # Handle both single and multiple GLT strings
    if isinstance(glt['glt_str'], list):
        # Multiple contrasts (e.g., from F-contrast)
        if fname is None:
            # Write each to separate file
            for i, (glt_str, name) in enumerate(zip(glt['glt_str'], glt['name'])):
                output_path = Path(f"{name}.txt")
                output_path.write_text(glt_str + "\n")
        else:
            # Write all to single file with comments
            output_path = Path(fname)
            content = []
            for glt_str, name in zip(glt['glt_str'], glt['name']):
                content.append(f"# {name}")
                content.append(glt_str)
            output_path.write_text("\n".join(content) + "\n")
    else:
        # Single contrast
        if fname is None:
            output_path = Path(f"{glt['name']}.txt")
        else:
            output_path = Path(fname)
        
        output_path.write_text(glt['glt_str'] + "\n")


# Implementation for dict-based contrasts (from contrast_weights output)
@to_glt.register(dict)
def _to_glt_dict(x: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Convert contrast dict to GLT format."""
    # Check if it's a contrast weights output
    if 'weights' in x and 'condnames' in x:
        weights = x['weights']
        condnames = x['condnames']
        name = x.get('name', 'contrast')
        
        if isinstance(weights, np.ndarray):
            if weights.ndim == 1:
                weights = weights.reshape(-1, 1)
            
            if weights.shape[1] > 1:
                # Multiple contrasts (e.g., F-contrast)
                glt_strs = []
                names = []
                
                for i in range(weights.shape[1]):
                    # Format each weight with 4 significant figures
                    weight_strs = []
                    for w, cond in zip(weights[:, i], condnames):
                        if abs(w) > 1e-10:  # Skip near-zero weights
                            # AFNI format: weight*condname
                            weight_str = f"{w:.4g}*{cond}"
                            weight_strs.append(weight_str)
                    
                    glt_str = " ".join(weight_strs)
                    glt_strs.append(glt_str)
                    names.append(f"GLT_{name}_{i+1}")
                
                return {
                    'glt_str': glt_strs,
                    'name': names,
                    'con': x
                }
            else:
                # Single contrast
                weight_strs = []
                for w, cond in zip(weights[:, 0], condnames):
                    if abs(w) > 1e-10:  # Skip near-zero weights
                        weight_str = f"{w:.4g}*{cond}"
                        weight_strs.append(weight_str)
                
                glt_str = " ".join(weight_strs)
                
                return {
                    'glt_str': glt_str,
                    'name': f"GLT_{name}",
                    'con': x
                }
    
    raise ValueError("Dict must contain 'weights' and 'condnames' keys")


# Helper to format AFNI commands
def format_afni_gltsym(glt: Dict[str, Any], label: Optional[str] = None) -> str:
    """Format GLT for AFNI -gltsym option.
    
    Parameters
    ----------
    glt : dict
        GLT dictionary from to_glt()
    label : str, optional
        Label for the GLT. If None, uses the name from glt
    
    Returns
    -------
    str
        Formatted string for AFNI -gltsym option
    
    Examples
    --------
    >>> glt = {'glt_str': '1*A -1*B', 'name': 'GLT_AvsB'}
    >>> print(format_afni_gltsym(glt))
    -gltsym 'SYM: 1*A -1*B' -glt_label 1 GLT_AvsB
    """
    if isinstance(glt['glt_str'], list):
        # Multiple GLTs
        lines = []
        for i, (glt_str, name) in enumerate(zip(glt['glt_str'], glt['name'])):
            lbl = label if label else name
            lines.append(f"-gltsym 'SYM: {glt_str}' -glt_label {i+1} {lbl}")
        return "\n".join(lines)
    else:
        # Single GLT
        lbl = label if label else glt['name']
        return f"-gltsym 'SYM: {glt['glt_str']}' -glt_label 1 {lbl}"


__all__ = [
    'to_glt',
    'write_glt',
    'format_afni_gltsym',
]