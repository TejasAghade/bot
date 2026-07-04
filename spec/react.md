# React rules

Binding conventions for React code in this project (`apps/web`, `apps/node/frontend`). Applies to `.ts` / `.tsx` files. Treat these as **gates** — code that violates them gets pushed back, not merged.

## Framework primitives first

Use the framework's own primitives before reaching for generic React patterns. The frameworks in this repo all provide higher-level abstractions for the common cases — using vanilla React when an idiomatic equivalent exists is the wrong default.

- **Both `apps/web` and `apps/node/frontend` use RR7 in SPA mode.** Route modules for organisation, `clientLoader` / `clientAction` for data, `<Link>` / `<Form>` / `useNavigate` for navigation, `useSearchParams` for URL state. **Never `useEffect(() => { fetchOrCall(...) }, [])`** — that's the anti-pattern RR7's data layer exists to replace, on either app.
- **`apps/web` data calls hit the server over HTTP** via the generated client in `apps/web/src/api/generated.ts`.
- **`apps/node/frontend` is wrapped by Wails.** Inside `clientLoader` / `clientAction`, backend calls go through Wails-generated TS bindings (`window.go.<package>.<method>`), not `fetch`. Wails events (`wails.events.on(...)`) drive cross-cutting state — but consume them via a React subscription hook, not in route modules.
- **React Three Fiber** → declarative scene, drei helpers, R3F hooks (`useFrame`, `useLoader`). Don't manage scene state via React state when R3F refs would do it.

Before writing a generic React pattern, check whether the framework you're in has its own idiomatic equivalent. If you don't know, **research it first** — don't reflex to vanilla React.

## File & module shape

- **Kebab-case filenames** for everything, including component files: `bag-stack-list.tsx`, not `BagStackList.tsx`. (Project-wide; restated.)
- **One component per file** is the default. Small presentational subcomponents used only inside the parent can co-locate.
- **PascalCase** for component names, **camelCase** for everything else.
- **Named exports** preferred. Default exports only when the framework requires them (RR7 route modules, lazy chunks).

## Component patterns

- **Function components only.** No class components, no `forwardRef` unless interop with a non-React library demands it.
- **TypeScript strict.** Every component's props typed via an `interface` or `type`. No `any` outside well-marked escape hatches with a one-line justification comment.
- **Destructure props in the signature.** No `props.foo` access patterns.
- **Children typed as `ReactNode`**, not `ReactElement` or `JSX.Element`, unless you have a reason.

## Componentize aggressively

Default to extraction. When a JSX block does a distinct visual or behavioral job — a header bar, a scene layer, a dialog body, a list row — it earns its own component file. The cost of one more small file is consistently lower than the cost of an inflated parent that you have to mentally diff to find the bit you care about.

- **Extract by default**; inline only when a subcomponent is used solely by its parent AND fits in a handful of lines AND has no independent reason to exist.
- **One component per file** (restated from "File & module shape" — the same rule, viewed from the extraction side).
- **Components compose**; the parent reads as a list of named children, not a wall of JSX.

The 60-line smell-trigger from `typescript.md` applies to React components too — a component pushing 80 lines (JSX inflates honestly, so the threshold is higher) is a strong signal that subtrees want to leave.

## Hooks (general)

- Rules of hooks apply: no conditional calls, top-level only. Biome's `useExhaustiveDependencies` lint stays on.
- **Custom hooks** start with `use`, return either a value or a typed tuple/object — never throw out of a hook.
- For `useEffect` specifically, see the dedicated section below — it is **not** a default tool.

## `useEffect` is an escape hatch, not a solution

This is the rule that fails most often in React codebases. Treat `useEffect` as a **last resort**, not a default. Most reasons to reach for it are wrong reasons.

Before writing a `useEffect`, work through this ladder:

1. **Is this derived state?** → `useMemo` or compute inline. Effects to keep state in sync with props are an anti-pattern.
2. **Is this server / network data?** → framework loader (`clientLoader` in RR7, Wails binding). Never `useEffect(() => { fetch... })`.
3. **Is this in response to a user action?** → handle it in the event handler, not in an effect.
4. **Is this DOM measurement?** → `useLayoutEffect`, or a ref + measurement-on-event.
5. **Is this synchronizing with an external system the framework doesn't know about?** (subscriptions, browser APIs not covered by hooks, third-party imperative libraries) → OK, `useEffect` may be the right tool.

Reaching for `useEffect` means option 5 is the answer **AND** you've explicitly ruled out 1-4. Question that conclusion. Research the alternatives. Only then write the effect.

**When `useEffect` does end up in code, it earns its place via a comment** — one short line above the effect explaining why none of options 1-4 applied. Then **flag it explicitly to the dev / user for review**: "I needed `useEffect` here, see comment, does this read right?"

Effects without a justification comment are a code smell on review. Effects without explicit review are a bigger one.

**Cleanup is mandatory.** Subscriptions, timers, observers — if the effect adds something, return a function that removes it.

**No effect-fired data mutations** that another effect listens to in the same component — render-loop bait. Compose into one effect or refactor.

## State

- **Local state first** (`useState` / `useReducer`).
- **Cross-component or app-wide state**: a designated store (zustand likely — not yet locked; revisit at first real need).
- **Server / network state**: framework's data layer (see "Framework primitives first" above). **Don't `fetch` inside components.**

## State ownership: lift state up, push subscriptions down

Two related rules, stated together because they pull in opposite directions and the balance matters:

- **Lift state up** to the lowest common ancestor of the components that read or write it. State that drives multiple subtrees lives there, not in one of the subtrees with prop-drilling cousins. At 4+ levels, lift to a store (per "State" above) instead of threading.
- **Push subscriptions down** to the leaf that actually consumes the value. A Wails event listener, a websocket subscription, a `window` event handler — wrap it in a small leaf component or hook so only that leaf re-renders on each tick. Parents stay still.

The combined effect: state ownership rises toward the root, side-effect subscriptions sink toward the leaves. Re-render scope stays tight; data flow stays readable.

Anti-pattern: a top-level component that owns a subscription, stores its value in state, and prop-drills it down through 3 layers. Every tick re-renders the whole subtree. Push the subscription into the leaf.

## Forms

- **`react-hook-form` + hand-rolled TypeBox resolver** for any form with more than one input. The resolver lives in `packages/shared-types/src/form-resolver.ts` and wraps `Value.Check` from `@sinclair/typebox/value`.
- **Never Zod.** Project-wide rule; see also the no-Zod feedback memory.
- TypeBox schemas shared with server schemas — import from `@acsm/shared-types`. The same schema validates the request on the server and the form on the client.
- Uncontrolled inputs only with library help. No raw `defaultValue` + refs.

## React Three Fiber

- **Declarative** — model the scene with `<mesh>`, `<group>`, drei helpers. Avoid imperative `scene.add(...)` unless wrapping a third-party Three.js library.
- **One `<Canvas>` per route.** Don't mount multiple canvases in one view; use one canvas with conditional sub-scenes.
- **Memoize Three.js objects** that are expensive to construct: `useMemo` for geometries, materials, loaded GLTFs.
- **Don't allocate inside render** without `useMemo` — geometries and materials should not be re-created every frame.
- **drei helpers** for common cases (`MapControls`, `PerspectiveCamera`, `useTexture`, `Environment`) over hand-rolled equivalents.

## Imports

- **ESM only** — no `require()` in app code.
- **Biome's `organizeImports`** runs on save; let it handle ordering. Don't manually reorder.
- **Type-only imports**: `import type` when importing types only — enforced by `verbatimModuleSyntax: true` in tsconfig.

## Performance

- **Don't memoize speculatively.** `React.memo` / `useMemo` / `useCallback` add complexity. Apply only when a profiler shows a real cost.
- **Avoid prop drilling beyond 3 levels.** At 4+, lift state up or add a context.
- **Lists** rendered from data must use a stable `key` from the data (id), never index unless the list is provably static.

## Styling

**shadcn/ui + Tailwind** for both `apps/web` and `apps/node/frontend`. Theme tokens carry Adani brand colors with a spacious feel (bumped radius + generous padding/gap defaults). Components are pulled into the repo via the shadcn CLI (full ownership, no runtime dep). Form components compose with `react-hook-form` + the TypeBox resolver (above).

- Don't reach for MUI / Chakra / Mantine. shadcn is the system.
- Don't write component CSS modules or styled-components alongside Tailwind. One styling mechanism.
- Custom one-off styles: Tailwind utility classes inline. If a pattern repeats 3+ times, extract a shadcn-component-style file.
