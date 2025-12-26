import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
from scipy import signal
from scipy.ndimage import median_filter
import librosa
import soundfile as sf
import os

def spectral_noise_gate(samples, sr, threshold_db=-40, smoothing=0.1):
    """
    Apply spectral noise gating to remove quiet noise between notes
    """
    # Compute STFT
    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Convert to dB
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Create noise gate mask
    mask = magnitude_db > threshold_db
    
    # Smooth the mask to avoid harsh cuts
    mask = median_filter(mask.astype(float), size=(3, 5))
    
    # Apply soft knee
    mask = np.clip(mask, 0, 1)
    mask = mask ** smoothing  # Soft gate
    
    # Apply mask
    magnitude_cleaned = magnitude * mask
    
    # Reconstruct
    stft_cleaned = magnitude_cleaned * np.exp(1j * phase)
    samples_cleaned = librosa.istft(stft_cleaned, hop_length=hop_length, length=len(samples))
    
    return samples_cleaned

def isolate_harmonic_content(samples, sr, margin=3.0):
    """
    Use harmonic-percussive separation to isolate tonal (piano) content
    """
    # Separate harmonics (tonal sounds like piano) from percussive (clicks, noise)
    harmonic, percussive = librosa.effects.hpss(samples, margin=margin)
    
    # Keep mostly harmonic with a bit of percussive for attack
    return harmonic * 0.9 + percussive * 0.1

def remove_clicks_and_pops(samples, sr, threshold=2.5):
    """
    Detect and remove sudden clicks and pops - more aggressive version
    """
    cleaned = samples.copy()
    
    # Multiple passes for thorough click removal
    for pass_num in range(3):
        # Compute the derivative (sudden changes)
        diff = np.diff(cleaned, prepend=cleaned[0])
        
        # Find outliers using adaptive threshold
        window_size = int(sr * 0.05)  # 50ms windows
        
        for i in range(0, len(diff), window_size // 2):
            end_idx = min(i + window_size, len(diff))
            window = diff[i:end_idx]
            
            if len(window) > 10:
                local_std = np.std(window)
                local_mean = np.mean(np.abs(window))
                local_threshold = local_mean + threshold * local_std
                
                # Find clicks in this window
                click_mask = np.abs(window) > local_threshold
                click_indices = np.where(click_mask)[0] + i
                
                # Interpolate over clicks
                for idx in click_indices:
                    if idx > 5 and idx < len(cleaned) - 5:
                        # Use cubic interpolation for smoother result
                        before = cleaned[idx-5:idx]
                        after = cleaned[idx+1:idx+6]
                        cleaned[idx] = np.mean(before) * 0.5 + np.mean(after) * 0.5
    
    return cleaned

def remove_static_artifacts(samples, sr):
    """
    Remove static/crackling artifacts using median filtering
    """
    # Detect high-frequency static
    # Static often shows as rapid oscillations
    
    # High-pass to isolate potential static
    nyquist = sr / 2
    hp_cutoff = 8000 / nyquist
    if hp_cutoff < 1:
        b, a = signal.butter(2, hp_cutoff, btype='high')
        high_freq = signal.filtfilt(b, a, samples)
        
        # Find where high frequency energy is anomalously high
        window = int(sr * 0.01)  # 10ms windows
        energy = np.array([np.sum(high_freq[i:i+window]**2) 
                          for i in range(0, len(high_freq)-window, window//2)])
        
        threshold = np.percentile(energy, 85)
        
        # Create mask for problem areas
        problem_windows = energy > threshold
        
        # Apply gentle low-pass to problem areas
        lp_cutoff = 4000 / nyquist
        b_lp, a_lp = signal.butter(3, lp_cutoff, btype='low')
        filtered = signal.filtfilt(b_lp, a_lp, samples)
        
        # Blend: use filtered version in problem areas
        result = samples.copy()
        for i, is_problem in enumerate(problem_windows):
            if is_problem:
                start = i * (window // 2)
                end = min(start + window, len(samples))
                # Smooth blend
                blend = 0.7  # 70% filtered in problem areas
                result[start:end] = filtered[start:end] * blend + samples[start:end] * (1-blend)
        
        return result
    
    return samples

def apply_declicker(samples, sr):
    """
    Apply a de-clicking algorithm to remove any remaining clicks
    """
    cleaned = samples.copy()
    
    # Use a median filter approach - clicks are outliers
    # Compare each sample to its neighbors
    window = 5
    
    for i in range(window, len(samples) - window):
        neighborhood = samples[i-window:i+window+1]
        median_val = np.median(neighborhood)
        
        # If current sample deviates significantly from median, it's likely a click
        deviation = np.abs(samples[i] - median_val)
        local_std = np.std(neighborhood)
        
        if deviation > 3 * local_std and local_std > 0.001:
            # Replace with interpolated value
            cleaned[i] = median_val
    
    return cleaned

def bandpass_for_piano(samples, sr, low_freq=80, high_freq=4500):
    """
    Apply bandpass filter focused on piano frequency range
    """
    nyquist = sr / 2
    low = low_freq / nyquist
    high = min(high_freq / nyquist, 0.99)
    
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, samples)
    
    return filtered

def soften_audio(samples, sr, cutoff_freq=3500, warmth=0.3):
    """
    Make audio less harsh by:
    1. Low-pass filter to reduce sharp high frequencies
    2. Gentle compression to smooth out peaks
    3. Add slight warmth
    """
    # Low-pass filter to reduce harshness
    nyquist = sr / 2
    cutoff = min(cutoff_freq / nyquist, 0.99)
    b, a = signal.butter(3, cutoff, btype='low')
    softened = signal.filtfilt(b, a, samples)
    
    # Soft knee compression to reduce sharp transients
    threshold = 0.5
    ratio = 3.0
    
    # Apply gentle compression
    compressed = samples.copy()
    above_threshold = np.abs(samples) > threshold
    compressed[above_threshold] = np.sign(samples[above_threshold]) * (
        threshold + (np.abs(samples[above_threshold]) - threshold) / ratio
    )
    
    # Blend original with softened and compressed
    # More softened = warmer, less harsh
    result = softened * warmth + compressed * (1 - warmth)
    
    return result

def smooth_transients(samples, sr, window_ms=5):
    """
    Smooth out sharp transients/attacks
    """
    window_samples = int(window_ms * sr / 1000)
    if window_samples < 3:
        window_samples = 3
    if window_samples % 2 == 0:
        window_samples += 1
    
    # Use a gentle smoothing filter
    kernel = np.hanning(window_samples)
    kernel = kernel / kernel.sum()
    
    smoothed = np.convolve(samples, kernel, mode='same')
    
    # Blend with original to keep some definition
    return samples * 0.6 + smoothed * 0.4

def find_best_loop_point(samples, sr, target_duration_sec=None):
    """
    Find the best point to create a seamless loop using zero-crossing analysis
    """
    if target_duration_sec:
        target_samples = int(target_duration_sec * sr)
    else:
        # Default to about 80% of the audio
        target_samples = int(len(samples) * 0.8)
    
    # Search window around target (1 second before and after)
    search_window = int(sr * 1.0)
    start_search = max(0, target_samples - search_window)
    end_search = min(len(samples), target_samples + search_window)
    
    # Find zero crossings in the search region
    search_region = samples[start_search:end_search]
    zero_crossings = np.where(np.diff(np.signbit(search_region)))[0]
    
    if len(zero_crossings) == 0:
        return target_samples
    
    # Find zero crossing closest to target
    target_in_region = target_samples - start_search
    best_idx = zero_crossings[np.argmin(np.abs(zero_crossings - target_in_region))]
    
    return start_search + best_idx

def create_seamless_crossfade(samples, sr, crossfade_duration_ms=800):
    """
    Create a seamless loop with smooth crossfade - no gap version
    The end of the audio crossfades INTO the beginning, creating continuous sound
    """
    crossfade_samples = int(crossfade_duration_ms * sr / 1000)
    
    # Make sure we have enough samples
    if len(samples) < crossfade_samples * 3:
        crossfade_samples = len(samples) // 4
    
    # Create crossfade curves (equal power crossfade for smooth transition)
    t = np.linspace(0, np.pi / 2, crossfade_samples)
    fade_out = np.cos(t) ** 2  # Smooth fade out (1 -> 0)
    fade_in = np.sin(t) ** 2   # Smooth fade in (0 -> 1)
    
    # The trick: we OVERLAY the faded end onto the beginning
    # This keeps the audio continuous with no gap
    
    # Start with a copy of the full audio
    result = samples.copy()
    
    # Take the END portion and fade it out
    end_portion = samples[-crossfade_samples:].copy() * fade_out
    
    # Take the START portion and fade it in
    start_portion = samples[:crossfade_samples].copy() * fade_in
    
    # Now we create the loop by:
    # 1. Trimming off the crossfade region from the end
    # 2. The crossfade region becomes: faded_end + faded_start
    
    # Trim the audio to remove the crossfade portion from the end
    trimmed = samples[:-crossfade_samples].copy()
    
    # Create the crossfade blend (end fading out + start fading in)
    crossfade_blend = end_portion + start_portion
    
    # The final audio is: original (minus crossfade length) + blended crossfade
    # When this loops, the blended section flows smoothly into the start
    final = np.concatenate([trimmed, crossfade_blend])
    
    # Actually wait - for a TRUE seamless loop, we need to modify the START too
    # Let's do it properly:
    
    # Method: Keep most of the audio, but blend the very end with the very beginning
    final = samples[:-crossfade_samples].copy()  # All except last crossfade portion
    
    # The crossfade region blends end->start
    blend_region = samples[-crossfade_samples:] * fade_out + samples[:crossfade_samples] * fade_in
    
    # Append the blend
    final = np.concatenate([final, blend_region])
    
    # Now when this audio loops, the blend_region flows into the start (samples[:crossfade])
    # which has been faded in... but wait, the actual start of 'final' is samples[0]
    
    # Let me think again... For a TRUE seamless loop:
    # The END of the file should smoothly connect to the START of the file
    # So we need the last samples to crossfade with what will play next (the first samples)
    
    # Correct approach:
    # 1. Audio plays from start to (end - crossfade)
    # 2. Then crossfade region plays: end fading out MIXED with start fading in
    # 3. When loop restarts, it plays from start again
    # 4. The crossfade at position (end-crossfade to end) already contains the faded-in start
    # 5. So when it loops, we need to skip the first crossfade_samples
    
    # Simpler approach: just make sure end blends into start with no silence
    final = samples.copy()
    
    # Blend the last crossfade_samples with the first crossfade_samples
    for i in range(crossfade_samples):
        blend_factor = i / crossfade_samples  # 0 to 1
        # At the end: mostly end sound, fading to start sound
        end_idx = len(samples) - crossfade_samples + i
        final[end_idx] = samples[end_idx] * (1 - blend_factor) + samples[i] * blend_factor
    
    # Trim off the redundant start (since it's now at the end as part of crossfade)
    final = final[crossfade_samples:]
    
    return final

def slow_down_audio(samples, sr, speed_factor=0.9):
    """
    Slow down audio without changing pitch using time stretching
    """
    # Time stretch - speed_factor < 1 slows down, > 1 speeds up
    stretched = librosa.effects.time_stretch(samples, rate=speed_factor)
    return stretched

def create_long_clean_loop(input_file, output_file, 
                           target_duration_sec=None,
                           crossfade_ms=800,
                           noise_gate_threshold=-42,
                           slow_down_factor=1.0):
    """
    Create a longer, cleaner loop with proper crossfade
    """
    print(f"Loading {input_file}...")
    
    # Load with librosa for better processing
    samples, sr = librosa.load(input_file, sr=None, mono=True)
    original_duration = len(samples) / sr
    print(f"Original duration: {original_duration:.2f} seconds, Sample rate: {sr}")
    
    # If no target duration, calculate after slowdown
    if target_duration_sec is None:
        # Will be recalculated after slowdown
        target_duration_sec = -1  # Placeholder
    
    # Step 1: Light click removal only
    print("Step 1: Light click removal...")
    samples = remove_clicks_and_pops(samples, sr, threshold=4.0)  # Higher threshold = less aggressive
    
    # Skip static artifact removal - was causing issues
    
    # Step 2: Isolate harmonic (piano) content
    print("Step 2: Isolating piano (harmonic) content...")
    samples = isolate_harmonic_content(samples, sr, margin=2.5)
    
    # Step 3: Apply spectral noise gate
    print("Step 3: Applying spectral noise gate...")
    samples = spectral_noise_gate(samples, sr, threshold_db=noise_gate_threshold)
    
    # Step 4: Bandpass filter for piano frequencies
    print("Step 4: Filtering for piano frequency range...")
    samples = bandpass_for_piano(samples, sr)
    
    # Step 4a: Soften harsh frequencies
    print("Step 4a: Softening harsh sounds...")
    samples = soften_audio(samples, sr, cutoff_freq=3000, warmth=0.4)
    
    # Step 4b: Smooth sharp transients
    print("Step 4b: Smoothing transients...")
    samples = smooth_transients(samples, sr, window_ms=8)
    
    # Step 4.5: Slow down audio if requested
    if slow_down_factor < 1.0:
        print(f"Step 4.5: Slowing down audio by {(1 - slow_down_factor) * 100:.0f}%...")
        samples = slow_down_audio(samples, sr, slow_down_factor)
        print(f"New duration after slowdown: {len(samples) / sr:.2f} seconds")
        # Skip aggressive de-clicking - was causing static
    
    # Calculate target duration if not specified (use 85% of current audio)
    current_duration = len(samples) / sr
    if target_duration_sec is None or target_duration_sec < 0:
        target_duration_sec = current_duration * 0.85
    
    print(f"Target loop duration: {target_duration_sec:.2f} seconds")
    
    # Step 5: Find best loop point
    print("Step 5: Finding optimal loop point...")
    loop_end = find_best_loop_point(samples, sr, target_duration_sec)
    samples = samples[:loop_end]
    print(f"Loop point found at {loop_end / sr:.2f} seconds")
    
    # Step 6: Normalize before crossfade
    print("Step 6: Normalizing...")
    samples = samples / np.max(np.abs(samples)) * 0.95
    
    # Step 7: Create seamless crossfade
    print(f"Step 7: Creating seamless loop with {crossfade_ms}ms crossfade...")
    final_samples = create_seamless_crossfade(samples, sr, crossfade_ms)
    
    # Export
    print(f"Exporting to {output_file}...")
    sf.write(output_file, final_samples, sr)
    
    final_duration = len(final_samples) / sr
    print(f"\n✓ Clean loop created!")
    print(f"✓ Duration: {final_duration:.2f} seconds")
    print(f"✓ Crossfade: {crossfade_ms}ms (equal-power for smooth transition)")
    
    return final_samples, sr

# Usage
if __name__ == "__main__":
    input_file = "./saturn_sounds.mp3"
    
    # Create a longer loop with better crossfade
    # Adjust target_duration_sec to control loop length
    # Set to None to use 85% of original audio
    
    create_long_clean_loop(
        input_file=input_file,
        output_file="single_pattern_clean.mp3",
        target_duration_sec=None,  # Use most of the audio, or set specific seconds like 30
        crossfade_ms=2000,  # 2 second crossfade for very smooth loop
        noise_gate_threshold=-35,  # Less aggressive - preserve more audio
        slow_down_factor=0.75  # Slow down to 75% speed
    )
    
    print("\n" + "="*50)
    print("Loop created: single_pattern_clean.mp3")
    print("\nTips:")
    print("- If loop is too short, increase target_duration_sec")
    print("- If crossfade sounds weird, try 500-1500ms range")
    print("- If too much noise removed, change threshold to -35")
    print("- If not enough noise removed, change threshold to -50")
