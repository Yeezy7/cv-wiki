export const sidebarSizing = {
  leftMin: 220,
  leftMax: 420,
  leftDefault: 300,
  leftCollapseAt: 168,
  rightMin: 220,
  rightMax: 360,
  rightDefault: 280,
  rightCollapseAt: 168,
} as const;

export const narrowViewportMediaQuery = '(max-width: 49.99rem)';

export const sidebarStorageKeys = {
  leftWidth: 'ai-left-sidebar-width',
  rightWidth: 'ai-right-sidebar-width',
  leftCollapsed: 'ai-left-sidebar-collapsed',
  rightCollapsed: 'ai-right-sidebar-collapsed',
} as const;
