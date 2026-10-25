---
name: fmrimod
description: A scientific methods atlas for executable fMRI modeling documentation.
colors:
  paper: "#fbfaf6"
  ink: "#202d31"
  muted-ink: "#526265"
  teal: "#216e65"
  teal-hover: "#164d47"
  rule: "#d5dcd6"
  code-surface: "#f5f5ef"
  night-paper: "#192526"
  night-ink: "#e4e9e2"
  night-muted: "#b0bfb7"
  night-teal: "#92c9bb"
  night-rule: "#435653"
  plot-teal: "#24786f"
  plot-terracotta: "#c26143"
  plot-blue: "#657aa4"
  plot-ochre: "#aa8b38"
  plot-plum: "#935d7a"
typography:
  display:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "3.5rem"
    fontWeight: 400
    lineHeight: 1.08
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "2.65rem"
    fontWeight: 400
    lineHeight: 1.13
    letterSpacing: "-0.025em"
  body:
    fontFamily: "Source Serif 4, Georgia, serif"
    fontSize: "17px"
    lineHeight: 1.7
  label:
    fontFamily: "-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
    fontSize: "0.79rem"
    lineHeight: 1.65
  code:
    fontFamily: "JetBrains Mono, Fira Code, ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"
    fontSize: "0.8rem"
    lineHeight: 1.6
rounded:
  square: "0"
  subtle: "0.15rem"
spacing:
  cell: "1.5rem"
  plate-gap: "2.2rem"
  reading-links-gap: "0.7rem 2rem"
components:
  code-block:
    backgroundColor: "{colors.code-surface}"
    textColor: "{colors.ink}"
    rounded: "{rounded.square}"
    padding: "0.9rem 1rem"
---

# Design System: fmrimod

## Overview

**Creative North Star: "Scientific Methods Atlas"**

A quiet, readable scientific publication presents executable analysis and its figures with equal care. Warm paper, restrained rules, and open figure plates leave the evidence room to speak.

Local serif reading type pairs with unobtrusive navigation and captions. Scientific plots retain their geometry and their own clear, restrained color vocabulary.

**Key Characteristics:**

- Open figure plates and flat, ruled surfaces.
- Serif reading and headings with sans-serif navigation and captions.
- Muted teal links and a measured scientific plot palette.
- Local assets and conventional Quarto navigation.

## Colors

The palette combines warm paper and green-black ink with muted teal and a complementary plot cycle. The sidecar tonal ramps are derived previews, not implemented or normative palette scales.

### Primary

- **Muted teal:** links and active navigation; a deeper teal indicates hover.

### Secondary

- **Terracotta:** a contrasting series in scientific plots, alongside teal, blue, ochre, and plum.

### Neutral

- **Warm paper and green-black ink:** the light reading surface and primary text.
- **Muted ink and pale rule:** supporting text and separators.
- **Night paper, ink, muted text, teal, and rule:** the corresponding dark-mode roles.

## Typography

**Reading and headings:** Source Serif 4, with Georgia and serif fallbacks. Regular and italic font files are local.

**Navigation and captions:** the system sans-serif stack. **Code:** the theme's monospace stack. Plot labels use DejaVu Sans.

The body uses a 17px root and 1.7 line height; article paragraphs are capped at 72ch. Titles use regular-weight serif type with balanced wrapping. Figure captions use 0.79rem sans-serif type and a 1.65 line height.

## Layout

Reading follows Quarto's article and navigation layout. Figure plates place a wide image beside a narrower note, separated by a 2.2rem gap and bounded by horizontal rules. At 991px and below the image and note stack. At 576px and below titles and code shrink, analysis routes stack, and documentation index tables become blocks. Wide output and reference tables scroll within their containers. Previous and next page labels wrap within links capped at half the navigation width. Figures preserve their aspect ratio.

## Elevation & Depth

Navigation and callouts are flat; borders and surface tones separate content. The theme defines subtle shadows, but the distinctive atlas plates and navigation do not rely on elevation.

## Shapes

Use flat edges and fine rules for plates, callouts, and source blocks. The shared theme retains a subtle 0.15rem radius for generic code surfaces; the documentation source block overrides it to square corners.

## Components

### Navigation

Quarto supplies the navbar, sidebar, search, theme switcher, and mobile navigation. Active sidebar and table-of-contents links use the accent, with weight or a rule instead of a filled pill. Visible focus receives a 2px accent outline with a 4px offset.

### Figure plates

Images are open and unboxed, with a concise adjacent or stacked note. At 576px and below, tutorial plot previews retain a 44rem width in a horizontally scrollable, keyboard-focusable region; captions retain the page measure. Quarto's lightbox provides the complete view. Homepage atlas figures instead fit the available width as overviews. Captions are left-aligned, muted, and set in sans-serif.

### Code and results

Source blocks use a thin border and a muted surface; output is distinguished by a left rule. Inline code has a quiet background. Code disables ligatures.

### Tables and callouts

Tables use sans-serif type, tabular figures, and horizontal rules. Callouts use a fine border, square corners, transparent interiors, and no shadow.

## Do's and Don'ts

### Do:

- **Do** preserve figure proportions and provide readable captions.
- **Do** keep navigation, links, and keyboard focus recognizable.
- **Do** use local font and figure assets.
- **Do** identify simulations and support scientific figures with executed code.

### Don't:

- **Don't** wrap scientific figures in decorative cards.
- **Don't** let wide tables force the page beyond the viewport.
- **Don't** interrupt reading with decorative animation.
