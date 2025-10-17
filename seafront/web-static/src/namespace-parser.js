"use strict";

/**
 * @typedef {Object} NamespaceNode
 * @property {string} name - Node name (e.g., "camera")
 * @property {string} path - Full path (e.g., "camera.main")
 * @property {ConfigItem[]} items - Config items at this level
 * @property {NamespaceNode[]} children - Child namespace nodes
 * @property {boolean} isExpanded - Whether the node is expanded in UI
 */

/**
 * Parse flat config items into hierarchical namespace tree
 * @param {ConfigItem[]} configItems - Flat array of config items
 * @returns {NamespaceNode[]} - Hierarchical tree structure
 */
export function parseConfigNamespaces(configItems) {
    const tree = [];
    const nodeMap = new Map(); // path -> NamespaceNode
    
    // Create root nodes and organize items
    for (const item of configItems) {
        const parts = item.handle.split('.');
        
        if (parts.length === 1) {
            // No namespace, add to root level
            let rootNode = nodeMap.get('');
            if (!rootNode) {
                rootNode = {
                    name: 'Root',
                    path: '',
                    items: [],
                    children: [],
                    isExpanded: true
                };
                nodeMap.set('', rootNode);
                tree.push(rootNode);
            }
            rootNode.items.push(item);
        } else {
            // Build namespace hierarchy
            let currentPath = '';
            let currentLevel = tree;
            
            for (let i = 0; i < parts.length - 1; i++) {
                const part = parts[i];
                const newPath = currentPath ? `${currentPath}.${part}` : part;
                
                let node = nodeMap.get(newPath);
                if (!node) {
                    node = {
                        name: part,
                        path: newPath,
                        items: [],
                        children: [],
                        isExpanded: false // Collapsed by default
                    };
                    nodeMap.set(newPath, node);
                    
                    // Find correct position in current level
                    const insertIndex = currentLevel.findIndex(n => n.name > part);
                    if (insertIndex === -1) {
                        currentLevel.push(node);
                    } else {
                        currentLevel.splice(insertIndex, 0, node);
                    }
                }
                
                currentLevel = node.children;
                currentPath = newPath;
            }
            
            // Add item to the deepest level
            const parentPath = parts.slice(0, -1).join('.');
            const parentNode = nodeMap.get(parentPath);
            if (parentNode) {
                parentNode.items.push(item);
            }
        }
    }
    
    return tree;
}

/**
 * Flatten namespace tree back to a filtered list of config items
 * @param {NamespaceNode[]} tree - Namespace tree
 * @param {string} filter - Filter string for handles
 * @returns {ConfigItem[]} - Filtered flat list
 */
export function flattenNamespaceTree(tree, filter = '') {
    /** @type {ConfigItem[]} */
    const items = [];
    
    /**
     * 
     * @param {NamespaceNode[]} nodes 
     */
    function traverse(nodes) {
        for (const node of nodes) {
            // Add items from this node
            for (const item of node.items) {
                if (!filter || item.handle.toLowerCase().includes(filter.toLowerCase())) {
                    items.push(item);
                }
            }
            
            // Traverse children
            traverse(node.children);
        }
    }
    
    traverse(tree);
    return items.sort((a, b) => a.handle.localeCompare(b.handle));
}

/**
 * Toggle expansion state of a namespace node
 * @param {NamespaceNode} node - Node to toggle
 */
export function toggleNamespaceExpansion(node) {
    node.isExpanded = !node.isExpanded;
}

/**
 * Expand all nodes in the tree
 * @param {NamespaceNode[]} tree - Namespace tree
 */
export function expandAllNamespaces(tree) {
    /**
     * 
     * @param {NamespaceNode[]} nodes 
     */
    function traverse(nodes) {
        for (const node of nodes) {
            node.isExpanded = true;
            traverse(node.children);
        }
    }
    traverse(tree);
}

/**
 * Collapse all nodes in the tree
 * @param {NamespaceNode[]} tree - Namespace tree
 */
export function collapseAllNamespaces(tree) {
    /**
     * 
     * @param {NamespaceNode[]} nodes 
     */
    function traverse(nodes) {
        for (const node of nodes) {
            node.isExpanded = false;
            traverse(node.children);
        }
    }
    traverse(tree);
}