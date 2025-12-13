# Documentation UI Improvement Proposal

**Version:** 1.0
**Date:** 2024-12-08
**Status:** Draft

---

## 1. Problem Summary

### Current State Analysis

Your MkDocs Material setup has a solid foundation but lacks refinement in several key areas:

#### Identified Issues

| Category | Problem | Impact |
|----------|---------|--------|
| **Content Width** | Default Material theme content stretches too wide on large screens | Reduced readability, eye strain |
| **Vertical Rhythm** | Inconsistent spacing between sections (`1rem` uniform) | Visual monotony, poor scannability |
| **Code Blocks** | No explicit overflow handling for long lines | Horizontal scroll or broken layouts |
| **Tables** | `width: 100%` forces tables to stretch awkwardly | Poor data presentation |
| **Typography** | Missing line-height and character-per-line limits | Harder to read long-form content |
| **Mobile** | No custom responsive adjustments | Cramped content on small screens |
| **Navigation** | Deep nesting (5+ levels) without visual distinction | Users get lost in hierarchy |

### Problem Patterns to Address

1. **Too-wide content**: Lines exceeding 80-90 characters reduce readability
2. **Lack of vertical rhythm**: Equal spacing doesn't create visual hierarchy
3. **Code block overflow**: Long code lines break layouts or require excessive scrolling
4. **Table responsiveness**: Wide tables don't adapt to narrow viewports
5. **Heading hierarchy weakness**: Insufficient size/weight differentiation
6. **Sidebar density**: Deep navigation trees become overwhelming
7. **Mobile navigation**: Collapsible menus need better touch targets

---

## 2. Reference Documentation Sites

### 2.1 Stripe Documentation
**URL:** https://stripe.com/docs

**Strengths:**
- Perfect content width (~720px) for optimal readability
- Three-column layout: nav | content | "On this page" TOC
- Code examples with language tabs and copy buttons
- Excellent dark mode implementation
- Request/response examples side-by-side

**Patterns to Adopt:**
- Fixed-width content container (not percentage-based)
- Sticky right-side TOC that highlights current section
- Code block styling with subtle backgrounds
- Breadcrumb navigation showing full path

### 2.2 Tailwind CSS Documentation
**URL:** https://tailwindcss.com/docs

**Strengths:**
- Clean, minimal design with excellent typography
- Smart search with instant results (Algolia)
- Collapsible sidebar sections with clear active states
- Inline code examples that are immediately usable
- Smooth transitions and micro-interactions

**Patterns to Adopt:**
- Sidebar group collapse/expand behavior
- Search overlay pattern
- Section headers with anchor links on hover
- Consistent spacing scale (4px base unit)

### 2.3 Vercel/Next.js Documentation
**URL:** https://nextjs.org/docs

**Strengths:**
- Modern, spacious layout with good whitespace
- Clear version selector in navigation
- Interactive code playgrounds
- "Was this helpful?" feedback at page bottom
- Excellent mobile experience

**Patterns to Adopt:**
- Version dropdown in header
- Feedback mechanism for content quality
- Mobile-first responsive approach
- Card-based navigation for landing pages

### 2.4 GitLab Documentation
**URL:** https://docs.gitlab.com

**Strengths:**
- Tier badges showing feature availability
- Topic-based organization with clear categories
- Good handling of technical/enterprise content depth
- Breadcrumbs with edit-on-GitLab links

**Patterns to Adopt:**
- Feature availability badges (relevant for trading system tiers)
- Topic taxonomy beyond simple hierarchy
- "Last updated" timestamps
- Contribution guidelines integration

### 2.5 Anthropic Documentation
**URL:** https://docs.anthropic.com

**Strengths:**
- Simple, focused design
- API reference with expandable sections
- Code examples in multiple languages
- Quick navigation between concepts and API

**Patterns to Adopt:**
- Expandable API reference sections
- Multi-language code tabs
- Clear concept/reference separation
- Minimal but effective styling

---

## 3. Layout & Wrapper Strategy

### 3.1 Desktop Layout (>1200px)

```
┌─────────────────────────────────────────────────────────────────┐
│ Header (fixed, 64px height)                                     │
├──────────┬───────────────────────────────────┬──────────────────┤
│ Sidebar  │ Main Content                      │ TOC             │
│ (280px)  │ (max-width: 800px, centered)      │ (220px)         │
│ fixed    │                                    │ sticky          │
│          │                                    │                 │
│          │                                    │                 │
└──────────┴───────────────────────────────────┴──────────────────┘
```

### 3.2 Tablet Layout (768px - 1200px)

```
┌─────────────────────────────────────────────────────────────────┐
│ Header (fixed, 64px)                          [hamburger menu]  │
├──────────┬──────────────────────────────────────────────────────┤
│ Sidebar  │ Main Content                                         │
│ (240px)  │ (max-width: 720px)                                   │
│ collaps. │ TOC moves to top of content as dropdown              │
└──────────┴──────────────────────────────────────────────────────┘
```

### 3.3 Mobile Layout (<768px)

```
┌─────────────────────────┐
│ Header [hamburger] [] │
├─────────────────────────┤
│ Breadcrumbs             │
├─────────────────────────┤
│ Main Content            │
│ (full width - 32px pad) │
│                         │
│ TOC: expandable section │
│ at top of content       │
└─────────────────────────┘
```

### 3.4 Recommended Breakpoints

```css
/* Spacing and breakpoint tokens */
:root {
  /* Breakpoints */
  --bp-mobile: 768px;
  --bp-tablet: 1200px;
  --bp-desktop: 1440px;

  /* Content widths */
  --content-max-width: 800px;
  --content-optimal-width: 720px;
  --sidebar-width: 280px;
  --toc-width: 220px;

  /* Spacing scale (4px base) */
  --space-1: 0.25rem;   /* 4px */
  --space-2: 0.5rem;    /* 8px */
  --space-3: 0.75rem;   /* 12px */
  --space-4: 1rem;      /* 16px */
  --space-5: 1.5rem;    /* 24px */
  --space-6: 2rem;      /* 32px */
  --space-8: 3rem;      /* 48px */
  --space-10: 4rem;     /* 64px */
}
```

---

## 4. Typography Guidelines

### 4.1 Base Typography

| Element | Size | Line Height | Weight | Max Width |
|---------|------|-------------|--------|-----------|
| Body | 16px | 1.6 | 400 | 75ch (~600px) |
| H1 | 32px | 1.2 | 700 | - |
| H2 | 24px | 1.3 | 600 | - |
| H3 | 20px | 1.4 | 600 | - |
| H4 | 16px | 1.5 | 600 | - |
| Code (inline) | 14px | 1.4 | 400 | - |
| Code (block) | 14px | 1.5 | 400 | 100% |

### 4.2 Heading Spacing

```css
/* Heading rhythm */
h1 {
  margin-top: 0;
  margin-bottom: var(--space-6);      /* 32px below */
}

h2 {
  margin-top: var(--space-8);         /* 48px above */
  margin-bottom: var(--space-4);      /* 16px below */
  padding-top: var(--space-4);
  border-top: 1px solid var(--md-default-fg-color--lightest);
}

h3 {
  margin-top: var(--space-6);         /* 32px above */
  margin-bottom: var(--space-3);      /* 12px below */
}

h4 {
  margin-top: var(--space-5);         /* 24px above */
  margin-bottom: var(--space-2);      /* 8px below */
}
```

### 4.3 Paragraph and List Spacing

```css
p {
  margin-bottom: var(--space-4);
  max-width: 75ch;                    /* Optimal line length */
}

ul, ol {
  margin-bottom: var(--space-4);
  padding-left: var(--space-5);
}

li {
  margin-bottom: var(--space-2);
}

li > ul, li > ol {
  margin-top: var(--space-2);
  margin-bottom: 0;
}
```

---

## 5. Code Block Guidelines

### 5.1 Styling

```css
/* Code block container */
.highlight {
  margin: var(--space-5) 0;
  border-radius: 8px;
  overflow: hidden;
  background: var(--md-code-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
}

/* Code content */
.highlight pre {
  margin: 0;
  padding: var(--space-4);
  overflow-x: auto;
  font-size: 14px;
  line-height: 1.5;
  tab-size: 2;
}

/* Line numbers */
.highlight .linenos {
  padding-right: var(--space-4);
  border-right: 1px solid var(--md-default-fg-color--lightest);
  color: var(--md-default-fg-color--light);
  user-select: none;
}
```

### 5.2 Overflow Handling

```css
/* Horizontal scroll with fade indicator */
.highlight {
  position: relative;
}

.highlight::after {
  content: '';
  position: absolute;
  top: 0;
  right: 0;
  bottom: 0;
  width: 40px;
  background: linear-gradient(90deg, transparent, var(--md-code-bg-color));
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
}

.highlight:hover::after {
  opacity: 1;
}

/* Long line indicator */
pre code {
  white-space: pre;
  word-wrap: normal;
}
```

### 5.3 Inline Code

```css
code:not(pre code) {
  padding: 0.15em 0.4em;
  border-radius: 4px;
  background: var(--md-code-bg-color);
  font-size: 0.875em;
  font-weight: 500;
}
```

---

## 6. Table Guidelines

### 6.1 Base Table Styling

```css
/* Table container for overflow */
.md-typeset__table {
  overflow-x: auto;
  margin: var(--space-5) 0;
}

table {
  width: auto;                        /* Don't force 100% */
  min-width: 50%;
  border-collapse: collapse;
  font-size: 0.9rem;
}

th, td {
  padding: var(--space-3) var(--space-4);
  text-align: left;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}

th {
  background: var(--md-default-fg-color--lightest);
  font-weight: 600;
  white-space: nowrap;
}

/* Zebra striping */
tbody tr:nth-child(even) {
  background: rgba(0, 0, 0, 0.02);
}

[data-md-color-scheme="slate"] tbody tr:nth-child(even) {
  background: rgba(255, 255, 255, 0.02);
}
```

### 6.2 Responsive Tables

```css
/* On mobile, allow horizontal scroll */
@media (max-width: 768px) {
  .md-typeset__table {
    display: block;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  table {
    min-width: 600px;                 /* Force scroll if needed */
  }
}
```

---

## 7. Navigation Improvements

### 7.1 Sidebar Enhancements

```css
/* Sidebar navigation */
.md-nav--primary .md-nav__item {
  padding: var(--space-1) 0;
}

/* Active section indicator */
.md-nav__link--active {
  font-weight: 600;
  color: var(--md-primary-fg-color);
  position: relative;
}

.md-nav__link--active::before {
  content: '';
  position: absolute;
  left: calc(-1 * var(--space-3));
  top: 0;
  bottom: 0;
  width: 3px;
  background: var(--md-primary-fg-color);
  border-radius: 2px;
}

/* Collapse deep sections by default */
.md-nav__item--nested > .md-nav {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.3s ease;
}

.md-nav__item--nested.md-nav__item--active > .md-nav {
  max-height: 1000px;
}
```

### 7.2 "On This Page" TOC

```css
/* Right-side TOC */
.md-sidebar--secondary {
  position: sticky;
  top: calc(var(--md-header-height) + var(--space-4));
  max-height: calc(100vh - var(--md-header-height) - var(--space-8));
  overflow-y: auto;
}

.md-sidebar--secondary .md-nav__link {
  font-size: 0.8rem;
  padding: var(--space-1) var(--space-2);
  border-left: 2px solid transparent;
  transition: all 0.2s;
}

.md-sidebar--secondary .md-nav__link:hover {
  border-left-color: var(--md-primary-fg-color);
}

.md-sidebar--secondary .md-nav__link--active {
  border-left-color: var(--md-primary-fg-color);
  background: var(--md-primary-fg-color--transparent);
}
```

---

## 8. Implementation Checklist

### Phase 1: Foundation (Priority: Critical)

- [ ] **Set content max-width to 800px**
  ```css
  .md-content__inner {
    max-width: 800px;
    margin: 0 auto;
  }
  ```

- [ ] **Implement spacing scale tokens**
  - Add CSS custom properties for consistent spacing
  - Replace all hardcoded `1rem` values with tokens

- [ ] **Fix line length for readability**
  ```css
  .md-typeset p, .md-typeset li {
    max-width: 75ch;
  }
  ```

- [ ] **Update heading hierarchy spacing**
  - H2: 48px top margin, 16px bottom
  - H3: 32px top margin, 12px bottom
  - H4: 24px top margin, 8px bottom

### Phase 2: Code & Tables (Priority: High)

- [ ] **Improve code block styling**
  - Add border and consistent border-radius
  - Implement horizontal scroll with gradient fade
  - Ensure copy button is always visible

- [ ] **Fix table overflow**
  - Remove `width: 100%` from tables
  - Add scrollable container wrapper
  - Implement mobile-responsive table behavior

- [ ] **Add syntax highlighting refinements**
  - Ensure dark mode colors are accessible
  - Add language label to code blocks

### Phase 3: Navigation (Priority: Medium)

- [ ] **Enhance sidebar navigation**
  - Add active state indicator (left border)
  - Implement collapsible sections for deep nesting
  - Reduce font size for deep-level items

- [ ] **Improve "On This Page" TOC**
  - Make sticky with proper offset
  - Add scroll-spy highlighting
  - Hide on mobile, show as dropdown

- [ ] **Add breadcrumb navigation**
  ```yaml
  # mkdocs.yml
  theme:
    features:
      - navigation.path  # Enable breadcrumbs
  ```

### Phase 4: Responsive (Priority: Medium)

- [ ] **Implement tablet breakpoint (768-1200px)**
  - Collapsible sidebar
  - TOC moves to content area
  - Adjust content padding

- [ ] **Implement mobile breakpoint (<768px)**
  - Full-width content with 16px padding
  - Hamburger menu for navigation
  - Expandable TOC at top of content

- [ ] **Touch-friendly targets**
  - Minimum 44px tap targets
  - Adequate spacing between links

### Phase 5: Polish (Priority: Low)

- [ ] **Add micro-interactions**
  - Smooth transitions on hover states
  - Scroll progress indicator
  - Anchor link hover reveal

- [ ] **Implement feedback mechanism**
  - "Was this page helpful?" at bottom
  - Link to edit on GitHub

- [ ] **Performance optimization**
  - Lazy-load images
  - Minimize custom CSS
  - Audit font loading

---

## 9. Updated CSS File

The following CSS should replace your current `extra.css`:

```css
/* ==========================================================================
   Ordinis Documentation - Enhanced Styles
   Version: 2.0
   ========================================================================== */

/* --------------------------------------------------------------------------
   1. Design Tokens
   -------------------------------------------------------------------------- */

:root {
  /* Brand Colors */
  --ordinis-primary: #3f51b5;
  --ordinis-primary-light: #7986cb;
  --ordinis-primary-dark: #303f9f;
  --ordinis-accent: #ff4081;
  --ordinis-success: #4caf50;
  --ordinis-warning: #ff9800;
  --ordinis-error: #f44336;

  /* Spacing Scale (4px base) */
  --space-1: 0.25rem;
  --space-2: 0.5rem;
  --space-3: 0.75rem;
  --space-4: 1rem;
  --space-5: 1.5rem;
  --space-6: 2rem;
  --space-8: 3rem;
  --space-10: 4rem;

  /* Layout */
  --content-max-width: 800px;
  --sidebar-width: 280px;
  --toc-width: 220px;

  /* Typography */
  --font-size-sm: 0.875rem;
  --font-size-base: 1rem;
  --font-size-lg: 1.125rem;
  --line-height-tight: 1.25;
  --line-height-normal: 1.6;
  --line-height-relaxed: 1.75;
}

/* --------------------------------------------------------------------------
   2. Content Layout
   -------------------------------------------------------------------------- */

/* Constrain content width for readability */
.md-content__inner {
  max-width: var(--content-max-width);
  margin-left: auto;
  margin-right: auto;
  padding: var(--space-6) var(--space-4);
}

/* Optimal line length for prose */
.md-typeset p,
.md-typeset li,
.md-typeset blockquote {
  max-width: 75ch;
}

/* --------------------------------------------------------------------------
   3. Typography
   -------------------------------------------------------------------------- */

.md-typeset {
  font-size: var(--font-size-base);
  line-height: var(--line-height-normal);
}

/* Heading Hierarchy */
.md-typeset h1 {
  font-size: 2rem;
  font-weight: 700;
  line-height: var(--line-height-tight);
  margin-top: 0;
  margin-bottom: var(--space-6);
  letter-spacing: -0.02em;
}

.md-typeset h2 {
  font-size: 1.5rem;
  font-weight: 600;
  line-height: 1.3;
  margin-top: var(--space-8);
  margin-bottom: var(--space-4);
  padding-top: var(--space-4);
  border-top: 1px solid var(--md-default-fg-color--lightest);
}

.md-typeset h3 {
  font-size: 1.25rem;
  font-weight: 600;
  line-height: 1.4;
  margin-top: var(--space-6);
  margin-bottom: var(--space-3);
}

.md-typeset h4 {
  font-size: 1rem;
  font-weight: 600;
  line-height: 1.5;
  margin-top: var(--space-5);
  margin-bottom: var(--space-2);
}

/* First heading after title doesn't need top border */
.md-typeset h1 + h2 {
  margin-top: var(--space-5);
  padding-top: 0;
  border-top: none;
}

/* Paragraph spacing */
.md-typeset p {
  margin-bottom: var(--space-4);
}

/* List spacing */
.md-typeset ul,
.md-typeset ol {
  margin-bottom: var(--space-4);
  padding-left: var(--space-5);
}

.md-typeset li {
  margin-bottom: var(--space-2);
}

.md-typeset li > ul,
.md-typeset li > ol {
  margin-top: var(--space-2);
  margin-bottom: 0;
}

/* --------------------------------------------------------------------------
   4. Code Blocks
   -------------------------------------------------------------------------- */

/* Code block container */
.md-typeset .highlight {
  margin: var(--space-5) 0;
  border-radius: 8px;
  border: 1px solid var(--md-default-fg-color--lightest);
  overflow: hidden;
}

.md-typeset .highlight pre {
  margin: 0;
  padding: var(--space-4);
  overflow-x: auto;
  font-size: 0.875rem;
  line-height: 1.6;
  tab-size: 2;
}

/* Horizontal scroll fade indicator */
.md-typeset .highlight {
  position: relative;
}

/* Inline code */
.md-typeset code:not(pre code) {
  padding: 0.125em 0.375em;
  border-radius: 4px;
  font-size: 0.875em;
  font-weight: 500;
  background: var(--md-code-bg-color);
  border: 1px solid var(--md-default-fg-color--lightest);
}

/* --------------------------------------------------------------------------
   5. Tables
   -------------------------------------------------------------------------- */

/* Table container */
.md-typeset__table {
  margin: var(--space-5) 0;
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
}

.md-typeset table:not([class]) {
  width: auto;
  min-width: 50%;
  display: table;
  border-collapse: collapse;
  font-size: var(--font-size-sm);
}

.md-typeset table:not([class]) th,
.md-typeset table:not([class]) td {
  padding: var(--space-3) var(--space-4);
  text-align: left;
  vertical-align: top;
  border-bottom: 1px solid var(--md-default-fg-color--lightest);
}

.md-typeset table:not([class]) th {
  font-weight: 600;
  white-space: nowrap;
  background: var(--md-default-fg-color--lightest);
}

/* Zebra striping */
.md-typeset table:not([class]) tbody tr:nth-child(even) {
  background: rgba(0, 0, 0, 0.02);
}

[data-md-color-scheme="slate"] .md-typeset table:not([class]) tbody tr:nth-child(even) {
  background: rgba(255, 255, 255, 0.02);
}

/* --------------------------------------------------------------------------
   6. Admonitions / Callouts
   -------------------------------------------------------------------------- */

.md-typeset .admonition {
  margin: var(--space-5) 0;
  border-radius: 8px;
  border-width: 1px;
  border-left-width: 4px;
}

.md-typeset .admonition-title {
  padding: var(--space-3) var(--space-4);
  font-weight: 600;
}

.md-typeset .admonition > p,
.md-typeset .admonition > ul,
.md-typeset .admonition > ol {
  padding: 0 var(--space-4) var(--space-4);
}

/* Custom governance admonition */
.md-typeset .admonition.governance {
  border-color: var(--ordinis-primary);
}

.md-typeset .admonition.governance > .admonition-title {
  background-color: rgba(63, 81, 181, 0.1);
}

.md-typeset .admonition.governance > .admonition-title::before {
  content: "governance";
}

/* --------------------------------------------------------------------------
   7. Navigation
   -------------------------------------------------------------------------- */

/* Sidebar navigation */
.md-nav--primary .md-nav__item {
  padding: var(--space-1) 0;
}

/* Active state indicator */
.md-nav__link--active {
  font-weight: 600;
  color: var(--md-primary-fg-color) !important;
}

/* TOC styling */
.md-nav--secondary {
  font-size: var(--font-size-sm);
}

.md-nav--secondary .md-nav__link {
  padding: var(--space-1) var(--space-2);
  border-left: 2px solid transparent;
  transition: border-color 0.2s, color 0.2s;
}

.md-nav--secondary .md-nav__link:hover,
.md-nav--secondary .md-nav__link--active {
  border-left-color: var(--md-primary-fg-color);
}

/* --------------------------------------------------------------------------
   8. Status Badges
   -------------------------------------------------------------------------- */

.version-badge {
  display: inline-block;
  padding: var(--space-1) var(--space-2);
  background-color: var(--ordinis-primary);
  color: white;
  border-radius: 4px;
  font-size: var(--font-size-sm);
  font-weight: 500;
}

.status-complete {
  color: var(--ordinis-success);
  font-weight: 600;
}

.status-in-progress {
  color: var(--ordinis-warning);
  font-weight: 600;
}

.status-pending {
  color: var(--ordinis-error);
  font-weight: 600;
}

/* --------------------------------------------------------------------------
   9. Responsive Adjustments
   -------------------------------------------------------------------------- */

/* Tablet */
@media (max-width: 1200px) {
  .md-content__inner {
    padding: var(--space-5) var(--space-4);
  }
}

/* Mobile */
@media (max-width: 768px) {
  :root {
    --space-6: 1.5rem;
    --space-8: 2rem;
  }

  .md-content__inner {
    padding: var(--space-4) var(--space-3);
  }

  .md-typeset h1 {
    font-size: 1.75rem;
  }

  .md-typeset h2 {
    font-size: 1.375rem;
    margin-top: var(--space-6);
  }

  .md-typeset h3 {
    font-size: 1.125rem;
  }

  /* Full-width tables on mobile with scroll */
  .md-typeset table:not([class]) {
    min-width: 100%;
  }
}

/* --------------------------------------------------------------------------
   10. Print Styles
   -------------------------------------------------------------------------- */

@media print {
  .md-sidebar,
  .md-header,
  .md-footer,
  .md-tabs,
  .md-search,
  .md-button {
    display: none !important;
  }

  .md-content {
    margin: 0 !important;
    padding: 0 !important;
    max-width: 100% !important;
  }

  .md-main__inner {
    margin: 0 !important;
  }

  h1, h2, h3, h4 {
    page-break-after: avoid;
  }

  table, figure, pre {
    page-break-inside: avoid;
  }

  a[href]::after {
    content: none !important;
  }
}

@page {
  size: letter;
  margin: 0.75in;
}

/* --------------------------------------------------------------------------
   11. Utility Classes
   -------------------------------------------------------------------------- */

.pdf-page-break {
  page-break-before: always;
}

.pdf-no-break {
  page-break-inside: avoid;
}

.doc-metadata {
  background-color: var(--md-default-fg-color--lightest);
  border-left: 4px solid var(--ordinis-primary);
  padding: var(--space-3) var(--space-4);
  margin-bottom: var(--space-5);
  font-size: var(--font-size-sm);
  border-radius: 0 4px 4px 0;
}

.doc-metadata dt {
  font-weight: 600;
  display: inline;
}

.doc-metadata dd {
  display: inline;
  margin-left: var(--space-2);
}

/* Mermaid diagrams */
.mermaid {
  text-align: center;
  margin: var(--space-5) 0;
}

/* Architecture diagrams */
pre.architecture {
  background-color: var(--md-code-bg-color);
  padding: var(--space-4);
  border-radius: 8px;
  overflow-x: auto;
  font-family: var(--md-code-font-family);
}
```

---

## 10. MkDocs Configuration Updates

Add these features to your `mkdocs.yml`:

```yaml
theme:
  features:
    # Existing features...
    - navigation.path          # Breadcrumbs
    - navigation.prune         # Optimize nav for large sites
    - toc.integrate            # Integrate TOC into sidebar on mobile
    - header.autohide          # Auto-hide header on scroll
    - content.tabs.link        # Link content tabs across page
```

---

## Summary

This proposal provides a systematic approach to improving your documentation UI:

1. **Content width** constrained to 800px for optimal readability
2. **Consistent spacing scale** using 4px base units
3. **Clear heading hierarchy** with distinct sizes and spacing
4. **Improved code blocks** with borders, scrolling, and copy buttons
5. **Responsive tables** that scroll horizontally when needed
6. **Enhanced navigation** with active states and sticky TOC
7. **Mobile-first responsive** design with appropriate breakpoints

The implementation checklist provides a prioritized path forward, starting with critical foundation work and progressing to polish items.
