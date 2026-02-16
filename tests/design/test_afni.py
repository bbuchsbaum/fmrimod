"""Tests for AFNI integration."""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from fmrimod.afni import to_glt, write_glt, format_afni_gltsym


class TestToGlt:
    """Test to_glt conversion function."""
    
    def test_to_glt_not_implemented(self):
        """Test to_glt with unsupported type."""
        with pytest.raises(NotImplementedError, match="to_glt not implemented"):
            to_glt("unsupported")
    
    def test_to_glt_single_contrast(self):
        """Test converting single contrast to GLT."""
        # Mock contrast weights output
        contrast = {
            'weights': np.array([1, -1, 0]),
            'condnames': ['condA', 'condB', 'condC'],
            'name': 'A_vs_B'
        }
        
        glt = to_glt(contrast)
        
        assert glt['glt_str'] == "1*condA -1*condB"
        assert glt['name'] == "GLT_A_vs_B"
        assert glt['con'] is contrast
    
    def test_to_glt_multiple_contrasts(self):
        """Test converting F-contrast (multiple contrasts) to GLT."""
        # Mock F-contrast weights output
        contrast = {
            'weights': np.array([
                [1, 0],
                [-1, 1],
                [0, -1]
            ]),
            'condnames': ['condA', 'condB', 'condC'],
            'name': 'main_effect'
        }
        
        glt = to_glt(contrast)
        
        assert isinstance(glt['glt_str'], list)
        assert len(glt['glt_str']) == 2
        assert glt['glt_str'][0] == "1*condA -1*condB"
        assert glt['glt_str'][1] == "1*condB -1*condC"
        assert glt['name'] == ['GLT_main_effect_1', 'GLT_main_effect_2']
        assert glt['con'] is contrast
    
    def test_to_glt_with_small_weights(self):
        """Test that near-zero weights are ignored."""
        contrast = {
            'weights': np.array([1.0, -1.0, 1e-12]),
            'condnames': ['condA', 'condB', 'condC'],
            'name': 'test'
        }
        
        glt = to_glt(contrast)
        
        # Small weight should be ignored
        assert glt['glt_str'] == "1*condA -1*condB"
    
    def test_to_glt_with_fractional_weights(self):
        """Test formatting of fractional weights."""
        contrast = {
            'weights': np.array([0.5, -0.5, 0.3333]),
            'condnames': ['condA', 'condB', 'condC'],
            'name': 'fractional'
        }
        
        glt = to_glt(contrast)
        
        # Check significant figures formatting
        assert "0.5*condA" in glt['glt_str']
        assert "-0.5*condB" in glt['glt_str']
        assert "0.3333*condC" in glt['glt_str']
    
    def test_to_glt_missing_keys(self):
        """Test error when required keys are missing."""
        with pytest.raises(ValueError, match="must contain 'weights' and 'condnames'"):
            to_glt({'name': 'test'})


class TestWriteGlt:
    """Test write_glt function."""
    
    def test_write_glt_single_to_file(self):
        """Test writing single GLT to specified file."""
        glt = {
            'glt_str': '1*condA -1*condB',
            'name': 'GLT_test',
            'con': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            fname = f.name
        
        try:
            write_glt(glt, fname)
            
            # Check file contents
            content = Path(fname).read_text()
            assert content.strip() == '1*condA -1*condB'
        finally:
            Path(fname).unlink()
    
    def test_write_glt_single_auto_filename(self):
        """Test writing single GLT with automatic filename."""
        glt = {
            'glt_str': '1*condA -1*condB',
            'name': 'GLT_test',
            'con': {}
        }
        
        # Write to current directory
        write_glt(glt)
        
        try:
            # Check file was created with correct name
            fname = Path('GLT_test.txt')
            assert fname.exists()
            
            content = fname.read_text()
            assert content.strip() == '1*condA -1*condB'
        finally:
            if fname.exists():
                fname.unlink()
    
    def test_write_glt_multiple_to_single_file(self):
        """Test writing multiple GLTs to single file."""
        glt = {
            'glt_str': ['1*condA -1*condB', '1*condB -1*condC'],
            'name': ['GLT_test_1', 'GLT_test_2'],
            'con': {}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            fname = f.name
        
        try:
            write_glt(glt, fname)
            
            # Check file contents
            content = Path(fname).read_text()
            lines = content.strip().split('\n')
            
            assert len(lines) == 4
            assert lines[0] == '# GLT_test_1'
            assert lines[1] == '1*condA -1*condB'
            assert lines[2] == '# GLT_test_2'
            assert lines[3] == '1*condB -1*condC'
        finally:
            Path(fname).unlink()
    
    def test_write_glt_multiple_auto_filenames(self):
        """Test writing multiple GLTs to separate files."""
        glt = {
            'glt_str': ['1*condA -1*condB', '1*condB -1*condC'],
            'name': ['GLT_test_1', 'GLT_test_2'],
            'con': {}
        }
        
        write_glt(glt)
        
        try:
            # Check both files were created
            fname1 = Path('GLT_test_1.txt')
            fname2 = Path('GLT_test_2.txt')
            
            assert fname1.exists()
            assert fname2.exists()
            
            assert fname1.read_text().strip() == '1*condA -1*condB'
            assert fname2.read_text().strip() == '1*condB -1*condC'
        finally:
            if fname1.exists():
                fname1.unlink()
            if fname2.exists():
                fname2.unlink()
    
    def test_write_glt_missing_glt_str(self):
        """Test error when glt_str is missing."""
        with pytest.raises(ValueError, match="must contain 'glt_str'"):
            write_glt({'name': 'test'})


class TestFormatAfniGltsym:
    """Test format_afni_gltsym function."""
    
    def test_format_single_glt(self):
        """Test formatting single GLT for AFNI command."""
        glt = {
            'glt_str': '1*condA -1*condB',
            'name': 'GLT_AvsB'
        }
        
        result = format_afni_gltsym(glt)
        expected = "-gltsym 'SYM: 1*condA -1*condB' -glt_label 1 GLT_AvsB"
        
        assert result == expected
    
    def test_format_single_glt_with_label(self):
        """Test formatting single GLT with custom label."""
        glt = {
            'glt_str': '1*condA -1*condB',
            'name': 'GLT_AvsB'
        }
        
        result = format_afni_gltsym(glt, label='CustomLabel')
        expected = "-gltsym 'SYM: 1*condA -1*condB' -glt_label 1 CustomLabel"
        
        assert result == expected
    
    def test_format_multiple_glts(self):
        """Test formatting multiple GLTs for AFNI command."""
        glt = {
            'glt_str': ['1*condA -1*condB', '1*condB -1*condC'],
            'name': ['GLT_test_1', 'GLT_test_2']
        }
        
        result = format_afni_gltsym(glt)
        lines = result.split('\n')
        
        assert len(lines) == 2
        assert lines[0] == "-gltsym 'SYM: 1*condA -1*condB' -glt_label 1 GLT_test_1"
        assert lines[1] == "-gltsym 'SYM: 1*condB -1*condC' -glt_label 2 GLT_test_2"


class TestIntegration:
    """Test integration with contrast weights."""
    
    def test_full_workflow(self):
        """Test complete workflow from contrast to GLT file."""
        # Simulate contrast weights output
        contrast = {
            'weights': np.array([0.5, 0.5, -0.5, -0.5]),
            'condnames': ['face_happy', 'face_sad', 'house_new', 'house_old'],
            'name': 'faces_vs_houses'
        }
        
        # Convert to GLT
        glt = to_glt(contrast)
        
        # Check GLT format
        assert '0.5*face_happy' in glt['glt_str']
        assert '0.5*face_sad' in glt['glt_str']
        assert '-0.5*house_new' in glt['glt_str']
        assert '-0.5*house_old' in glt['glt_str']
        
        # Write to file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            fname = f.name
        
        try:
            write_glt(glt, fname)
            
            # Verify file contents
            content = Path(fname).read_text()
            assert '0.5*face_happy' in content
            assert '-0.5*house_old' in content
            
            # Check AFNI command format
            cmd = format_afni_gltsym(glt)
            assert 'GLT_faces_vs_houses' in cmd
            assert glt['glt_str'] in cmd
        finally:
            Path(fname).unlink()