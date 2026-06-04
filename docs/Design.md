---
name: Parlor Soft-Goods
colors:
  surface: '#fbf9f1'
  surface-dim: '#dcdad2'
  surface-bright: '#fbf9f1'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f5f4ec'
  surface-container: '#f0eee6'
  surface-container-high: '#eae8e0'
  surface-container-highest: '#e4e3db'
  on-surface: '#1b1c17'
  on-surface-variant: '#41484e'
  inverse-surface: '#30312c'
  inverse-on-surface: '#f3f1e9'
  outline: '#71787f'
  outline-variant: '#c0c7cf'
  surface-tint: '#1c648e'
  primary: '#1c648e'
  on-primary: '#ffffff'
  primary-container: '#7cb9e8'
  on-primary-container: '#00496d'
  inverse-primary: '#90cdfd'
  secondary: '#a43b32'
  on-secondary: '#ffffff'
  secondary-container: '#fd7d6f'
  on-secondary-container: '#711611'
  tertiary: '#246d00'
  on-tertiary: '#ffffff'
  tertiary-container: '#6dc747'
  on-tertiary-container: '#18501c648e00'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#cae6ff'
  primary-fixed-dim: '#90cdfd'
  on-primary-fixed: '#001e30'
  on-primary-fixed-variant: '#004b70'
  secondary-fixed: '#ffdad5'
  secondary-fixed-dim: '#ffb4aa'
  on-secondary-fixed: '#410001'
  on-secondary-fixed-variant: '#84241d'
  tertiary-fixed: '#9cf973'
  tertiary-fixed-dim: '#81dc5a'
  on-tertiary-fixed: '#062100'
  on-tertiary-fixed-variant: '#195200'
  background: '#fbf9f1'
  on-background: '#1b1c17'
  surface-variant: '#e4e3db'
typography:
  headline-lg:
    fontFamily: Quicksand
    fontSize: 40px
    fontWeight: '700'
    lineHeight: 48px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Quicksand
    fontSize: 32px
    fontWeight: '700'
    lineHeight: 38px
  headline-md:
    fontFamily: Quicksand
    fontSize: 28px
    fontWeight: '600'
    lineHeight: 36px
  body-lg:
    fontFamily: Plus Jakarta Sans
    fontSize: 18px
    fontWeight: '400'
    lineHeight: 28px
  body-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
  label-md:
    fontFamily: Plus Jakarta Sans
    fontSize: 14px
    fontWeight: '600'
    lineHeight: 20px
    letterSpacing: 0.01em
  label-sm:
    fontFamily: Plus Jakarta Sans
    fontSize: 12px
    fontWeight: '700'
    lineHeight: 16px
rounded:
  sm: 0.25rem
  DEFAULT: 0.5rem
  md: 0.75rem
  lg: 1rem
  xl: 1.5rem
  full: 9999px
spacing:
  base: 8px
  margin-mobile: 20px
  margin-desktop: 64px
  gutter: 24px
  container-max: 1200px
---

## Brand & Style

The design system moves away from technical coldness toward a "Parlor" aesthetic—a warm, domestic, and joyful environment. It targets a community-driven audience looking for comfort and approachability. The UI should evoke a sense of a "digital quilt": handcrafted, soft, and inviting.

The style is a blend of **Minimalism** and **Tactile Design**. It utilizes a light, airy foundation with "squishy" interactive elements. The brand personality is optimistic, gentle, and legible, replacing high-tech precision with human-centric warmth. Visuals are defined by high-contrast pastel accents against cream backgrounds, ensuring the "happy" vibe remains functional and accessible.

## Colors

The palette is a curated selection of "Sunday Morning" pastels. The foundation is a warm off-white (Cream), which provides a softer base than pure white. 

- **Primary (Pastel Blue):** Used for main actions and trust-building elements.
- **Secondary (Coral):** Used for highlights and important notifications.
- **Tertiary (Pastel Green):** Used for success states and growth indicators.
- **Accents (Lavender, Yellow, Pink):** Used for categorization and playful "quilt" patterns.

To maintain accessibility, all text is rendered in a deep charcoal-brown rather than pure black, ensuring high contrast against the pastel backgrounds while maintaining the warm tone.

## Typography

This design system prioritizes rounded, geometric letterforms to reinforce the friendly narrative. **Quicksand** is the primary choice for headlines, offering a soft terminal that feels organic and approachable.

**Plus Jakarta Sans** is utilized for body text and labels. It provides a contemporary, clean structure that remains highly legible at smaller sizes, ensuring that the "soft" aesthetic does not compromise functional reading. Line heights are generous to prevent the UI from feeling cramped or "technical."

## Layout & Spacing

The layout follows a **Fluid Grid** model with an emphasis on "breathing room." Elements are spaced using an 8px base unit, but padding is intentionally generous to create a relaxed, unhurried pace.

On Desktop, a 12-column grid is used with wide 64px margins. On Mobile, the layout collapses to a single column with 20px margins. A signature element of this system is the "Quilt Block" layout: content is often grouped in uneven, slightly offset containers that mimic the stitching of a handmade quilt, breaking the rigidity of standard enterprise grids.

## Elevation & Depth

Depth is achieved through **Tonal Layers** and **Ambient Shadows**. Instead of harsh black shadows, this design system uses soft, diffused shadows tinted with the primary or secondary colors (e.g., a soft blue shadow under a blue button).

The hierarchy is flat but tactile. Surfaces use subtle 1px inner borders in a slightly darker shade of the background color to simulate the "edge" of a fabric or paper. When an element is hovered, it should "lift" slightly and increase its shadow diffusion, creating a "squishy" response to user interaction.

## Shapes

The shape language is consistently **Rounded**. There are no sharp corners in the design system. 

- **Standard Containers:** Use 0.5rem (8px) corners.
- **Interactive Elements:** Buttons and Inputs use 1rem (16px) or full "Pill" shapes to encourage clicking.
- **Quilt Patterns:** Backgrounds feature repeating geometric shapes (diamonds, soft triangles) in low-opacity pastel tones to add texture without distracting from content.

## Components

- **Buttons:** High-contrast pastel backgrounds with deep-colored text. They use a "press" effect where the element physically shifts down 2px on click.
- **Cards:** Defined by the "Cream" background with a soft, tinted shadow. Cards often feature a 4px "stitch" border (dashed) in a pastel accent color.
- **Inputs:** Soft-rounded containers with a subtle cream-to-white gradient. Focus states are indicated by a 2px solid pastel blue border.
- **Chips:** Small, pill-shaped tags using the full pastel palette to categorize content.
- **Progress Bars:** Thick, rounded tracks with a "bubble" style filler in pastel green.
- **Lists:** Separated by soft, dotted lines rather than solid rules, maintaining the light and "stitched" feel of the system.