import cv2
import numpy as np
from midiutil import MIDIFile


def show_and_destroy(img, window=''):
    cv2.imshow(window, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def match_pattern_unique_tempo(src, pattern, bnry):
    w, h = pattern.shape[::-1]
    res = cv2.matchTemplate(src, pattern, cv2.TM_CCOEFF_NORMED)
    *_, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(bnry, top_left, bottom_right, TEMPO_LABEL, 1)
    return bnry


def get_loc_for_threshold(src, pattern, threshold):
    res = cv2.matchTemplate(src, pattern, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    return loc


def match_pattern_threshold_tempo(src, pattern, bnry, threshold=0.9):
    w, h = pattern.shape[::-1]
    loc = get_loc_for_threshold(src, pattern, threshold)
    b = bnry.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(b, pt, (pt[0] + w, pt[1] + h), TEMPO_LABEL, 1)
    return b


def match_pattern_threshold_pitch(src, pattern, bnry, threshold=0.9):
    w, h = pattern.shape[::-1]
    loc = get_loc_for_threshold(src, pattern, threshold)
    b = bnry.copy()
    for pt in zip(*loc[::-1]):
        cv2.circle(b, (pt[0] + w // 2, pt[1] + h // 2), 0, TONE_LABEL)
    return b


# Global variables
TEMPO_LABEL = 1
TONE_LABEL = 254

# Image reading
source = cv2.imread('sample-a.png', cv2.IMREAD_GRAYSCALE)
width, height = source.shape[::-1]

print(f"\nw = {width}, h = {height}")

# Image negative
negative = cv2.bitwise_not(source)

# Threshold
binary = cv2.adaptiveThreshold(negative, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
show_and_destroy(binary, 'Binary and negative image')

# Copies for image processing
bw_hor, bw_ver = binary.copy(), binary.copy()
notes_rows, notes_cols = binary.shape

# Structuring element for lines isolation
hor_size = notes_cols // 30
hor_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_size, 1))

# Isolating horizontal lines
bw_hor = cv2.erode(bw_hor, hor_structure)
bw_hor = cv2.dilate(bw_hor, hor_structure)

# Structuring element for lines removal
ver_size = notes_rows // 30
ver_struct = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_size))

# Removing horizontal lines
bw_ver = cv2.erode(bw_ver, ver_struct)
bw_ver = cv2.dilate(bw_ver, ver_struct)

# Making bw_ver not negative again
bw_ver = cv2.bitwise_not(bw_ver)

# Notes edges
edges = cv2.adaptiveThreshold(bw_ver, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, -2)

# Notes edges dilatation
kernel = np.ones((2, 2), np.uint8)
edges = cv2.dilate(edges, kernel)

# Creating smoothed image for element searching
smoothed = bw_ver.copy()
smoothed = cv2.blur(smoothed, (2, 2))
notes_rows, notes_cols = np.where(edges != 0)
bw_ver[notes_rows, notes_cols] = smoothed[notes_rows, notes_cols]

show_and_destroy(bw_ver, 'bw_ver after smoothing')

binary = cv2.bitwise_not(binary)

# Image for tempo mapping
tempo_map_img = binary.copy()

# Analyze tempo data
TEMPO_LABEL += 1

# 8
eighteen_note = cv2.imread('symbols/8.jpg', 0)
tempo_map_img = match_pattern_threshold_tempo(bw_ver, eighteen_note, tempo_map_img, threshold=0.90)

TEMPO_LABEL += 1

# 4d
dotted_quarter_note = cv2.imread('symbols/4d-up.png', 0)
tempo_map_img = match_pattern_threshold_tempo(bw_ver, dotted_quarter_note, tempo_map_img, threshold=0.95)

inverted_dotted_quarter_note = cv2.imread('symbols/4d-down.png', 0)
tempo_map_img = match_pattern_threshold_tempo(bw_ver, inverted_dotted_quarter_note, tempo_map_img, threshold=0.95)

TEMPO_LABEL += 1

# 4
quarter_note = cv2.imread('symbols/4.png', 0)
tempo_map_img = match_pattern_threshold_tempo(bw_ver, quarter_note, tempo_map_img, threshold=0.85)

inverted_quarter_note = cv2.rotate(quarter_note, cv2.ROTATE_180)
tempo_map_img = match_pattern_threshold_tempo(bw_ver, inverted_quarter_note, tempo_map_img, threshold=0.85)

TEMPO_LABEL += 1

# 2
half_note = cv2.imread('symbols/2.jpg', 0)
tempo_map_img = match_pattern_threshold_tempo(bw_ver, half_note, tempo_map_img, threshold=0.85)

# Analyze pentagram
pentagram_img = bw_hor.copy()

rows_histogram = np.zeros((height, 1))
for i in range(height):
    zeros = cv2.bitwise_not(bw_hor)[i, :] == 0
    rows_histogram[i] = np.sum(zeros)

tmp = []
for i in range(height):
    if rows_histogram[i] > int(0.8 * width):
        tmp.append(i)

tmp = np.asarray(tmp)

# Estimate space between lines using average
s = 0
for i in range(np.size(tmp) - 1):
    s += tmp[i + 1] - tmp[i]
LINE_SPACING = s / 4
G4_LINE = tmp[3]
C4_LINE = G4_LINE + 2 * LINE_SPACING

HEIGHTS = np.asarray([
    C4_LINE - i * (LINE_SPACING / 2) for i in range(24)
    if C4_LINE - i * (LINE_SPACING / 2) >= 0
])

"""
Tempo-map creation
    Labels      Description     Hash
    2           8               0.5
    3           4d              1.5
    4           4               1.0
    5           2               2.0
"""

# Create tempo map sorting notes by horizontal position
tmp = {}
for label in range(2, 6):
    for i in range(height):
        for j in range(width):
            if tempo_map_img[i, j] == label:
                cv2.floodFill(tempo_map_img, None, (j, i), 255)
                tmp[(j, i)] = label  # j first for sort by column position

tempo_ij = list(tmp)
tempo_ij.sort()
sorted_tmp = {}
for ij in tempo_ij:
    sorted_tmp[ij] = tmp[ij]

TEMPO_MAP = list(sorted_tmp.values())

print(f"\nTEMPO MAP: {TEMPO_MAP}")

# Create pitch map
tmp.clear()
sorted_tmp.clear()

tones_map_img = bw_ver.copy()

# Full head matching
full_head_pattern = cv2.imread('symbols/4-4d-8H.png', 0)
tones_map_img = match_pattern_threshold_pitch(tones_map_img, full_head_pattern, bw_ver, threshold=0.9)

for i in range(height):
    for j in range(width):
        if tones_map_img[i, j] == TONE_LABEL:
            cv2.floodFill(tones_map_img, None, (j, i), 0)
            tmp[(j, i)] = i

# Open-head matching
open_head_pattern = cv2.imread('symbols/2H.png', 0)
tones_map_img = match_pattern_threshold_pitch(tones_map_img, open_head_pattern, bw_ver, threshold=0.86)

for i in range(height):
    for j in range(width):
        if tones_map_img[i, j] == TONE_LABEL:
            cv2.floodFill(tones_map_img, None, (j, i), 255)
            tmp[(j, i)] = i

# Sort pitches
tones_ij = list(tmp)
tones_ij.sort()
sorted_tmp = {}
for symbol in tones_ij:
    sorted_tmp[symbol] = tmp[symbol]

TONES_MAP = list(sorted_tmp.values())

print(f"\nPITCH MAP: {TONES_MAP}")

print(f"\nHEIGHTS: {HEIGHTS}")

STEPS = []

for i, pitch in enumerate(TONES_MAP):
    nearest = (np.abs(pitch - HEIGHTS)).argmin()
    STEPS.append(nearest)

print(f"\nSTEPS: {STEPS}")

# Verifying if there is the same amount of tempos and tones
assert len(TONES_MAP) == len(TEMPO_MAP), "Different amount of tempos and notes"

# Creating MIDI file from mapped data
midi = MIDIFile(numTracks=1)
midi.addTempo(track=0, time=0, tempo=80)

# Hash for tempos and tones
hsh_tempo = {2: 0.5, 3: 1.5, 4: 1., 5: 2.}

c4_to_b4 = np.array([60, 62, 64, 65, 67, 69, 71])
c5_to_b5 = c4_to_b4 + 12

c4_to_b4_map = {i: c4_to_b4[i] for i in range(np.size(c4_to_b4))}
c5_to_b5_map = {i + 7: c5_to_b5[i] for i in range(np.size(c5_to_b5))}
c4_to_b5_map = {**c4_to_b4_map, **c5_to_b5_map}

# Inserting midi data into midi file
time = 0
for i, tone in enumerate(TONES_MAP):
    midi.addNote(
        track=0,
        channel=0,
        time=time,
        duration=hsh_tempo[TEMPO_MAP[i]],
        pitch=c4_to_b5_map[STEPS[i]],
        volume=64
    )
    time += hsh_tempo[TEMPO_MAP[i]]

with open("music.mid", "wb") as output_file:
    midi.writeFile(output_file)
