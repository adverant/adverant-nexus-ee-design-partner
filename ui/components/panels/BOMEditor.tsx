"use client";

/**
 * BOM Editor - Bill of Materials CRUD Interface
 *
 * Full-featured editor for managing component lists with
 * add, edit, delete, import/export capabilities.
 */

import { useState, useMemo, useCallback } from "react";
import { cn } from "@/lib/utils";
import {
  Plus,
  Trash2,
  Download,
  Upload,
  Edit2,
  Save,
  X,
  Search,
  ExternalLink,
  Package,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

// ============================================================================
// Types
// ============================================================================

export interface BOMItem {
  id: string;
  reference: string;
  partNumber: string;
  manufacturer: string;
  description: string;
  quantity: number;
  footprint: string;
  value?: string;
  price?: number;
  supplier?: string;
  supplierPartNumber?: string;
  inStock?: boolean;
  datasheetUrl?: string;
}

interface BOMEditorProps {
  /** Project ID for API calls */
  projectId: string;
  /** Initial BOM items */
  initialItems?: BOMItem[];
  /** Optional class name */
  className?: string;
  /** Callback when BOM changes */
  onChange?: (items: BOMItem[]) => void;
  /** Callback to save BOM */
  onSave?: (items: BOMItem[]) => Promise<void>;
}

type SortField = keyof BOMItem;
type SortDirection = "asc" | "desc";

// ============================================================================
// Edit Modal Component (Simple implementation without radix-ui)
// ============================================================================

interface EditModalProps {
  item: BOMItem | null;
  isNew: boolean;
  onSave: (item: BOMItem) => void;
  onCancel: () => void;
}

function EditModal({ item, isNew, onSave, onCancel }: EditModalProps) {
  const [formData, setFormData] = useState<BOMItem>(
    item || {
      id: crypto.randomUUID(),
      reference: "",
      partNumber: "",
      manufacturer: "",
      description: "",
      quantity: 1,
      footprint: "",
      value: "",
    }
  );

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSave(formData);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Overlay */}
      <div className="absolute inset-0 bg-black/60" onClick={onCancel} />

      {/* Modal */}
      <div className="relative bg-slate-800 rounded-lg shadow-xl p-6 w-full max-w-lg border border-slate-700">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-white">
            {isNew ? "Add Component" : "Edit Component"}
          </h2>
          <button
            onClick={onCancel}
            className="p-1 hover:bg-slate-700 rounded"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Reference</label>
              <input
                type="text"
                value={formData.reference}
                onChange={(e) => setFormData({ ...formData, reference: e.target.value })}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
                placeholder="U1, R1, C1..."
                required
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Quantity</label>
              <input
                type="number"
                value={formData.quantity}
                onChange={(e) =>
                  setFormData({ ...formData, quantity: parseInt(e.target.value) || 1 })
                }
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
                min="1"
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Part Number</label>
            <input
              type="text"
              value={formData.partNumber}
              onChange={(e) => setFormData({ ...formData, partNumber: e.target.value })}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
              placeholder="STM32G431CBT6"
              required
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Manufacturer</label>
            <input
              type="text"
              value={formData.manufacturer}
              onChange={(e) => setFormData({ ...formData, manufacturer: e.target.value })}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
              placeholder="STMicroelectronics"
            />
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Description</label>
            <input
              type="text"
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
              placeholder="32-bit ARM Cortex-M4 MCU"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Footprint</label>
              <input
                type="text"
                value={formData.footprint}
                onChange={(e) => setFormData({ ...formData, footprint: e.target.value })}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
                placeholder="LQFP-48"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Value</label>
              <input
                type="text"
                value={formData.value || ""}
                onChange={(e) => setFormData({ ...formData, value: e.target.value })}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
                placeholder="100nF, 10k..."
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm text-slate-400 mb-1">Price (USD)</label>
              <input
                type="number"
                step="0.01"
                value={formData.price || ""}
                onChange={(e) =>
                  setFormData({ ...formData, price: parseFloat(e.target.value) || undefined })
                }
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
                placeholder="2.50"
              />
            </div>
            <div>
              <label className="block text-sm text-slate-400 mb-1">Supplier</label>
              <input
                type="text"
                value={formData.supplier || ""}
                onChange={(e) => setFormData({ ...formData, supplier: e.target.value })}
                className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
                placeholder="DigiKey, Mouser..."
              />
            </div>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Datasheet URL</label>
            <input
              type="url"
              value={formData.datasheetUrl || ""}
              onChange={(e) => setFormData({ ...formData, datasheetUrl: e.target.value })}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:border-primary-500 focus:outline-none"
              placeholder="https://..."
            />
          </div>

          <div className="flex justify-end gap-2 pt-4">
            <button
              type="button"
              onClick={onCancel}
              className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
            >
              <Save className="w-4 h-4" />
              {isNew ? "Add" : "Save"}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function BOMEditor({
  projectId,
  initialItems = [],
  className,
  onChange,
  onSave,
}: BOMEditorProps) {
  const [items, setItems] = useState<BOMItem[]>(initialItems);
  const [searchQuery, setSearchQuery] = useState("");
  const [sortField, setSortField] = useState<SortField>("reference");
  const [sortDirection, setSortDirection] = useState<SortDirection>("asc");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [editingItem, setEditingItem] = useState<BOMItem | null>(null);
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Filter and sort items
  const filteredItems = useMemo(() => {
    let result = items;

    // Search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      result = result.filter(
        (item) =>
          item.reference.toLowerCase().includes(query) ||
          item.partNumber.toLowerCase().includes(query) ||
          item.manufacturer.toLowerCase().includes(query) ||
          item.description.toLowerCase().includes(query)
      );
    }

    // Sort
    result = [...result].sort((a, b) => {
      const aVal = a[sortField] ?? "";
      const bVal = b[sortField] ?? "";
      const comparison = String(aVal).localeCompare(String(bVal), undefined, { numeric: true });
      return sortDirection === "asc" ? comparison : -comparison;
    });

    return result;
  }, [items, searchQuery, sortField, sortDirection]);

  // Calculate totals
  const totals = useMemo(() => {
    const totalQuantity = items.reduce((sum, item) => sum + item.quantity, 0);
    const totalPrice = items.reduce(
      (sum, item) => sum + (item.price || 0) * item.quantity,
      0
    );
    const uniqueParts = new Set(items.map((item) => item.partNumber)).size;
    return { totalQuantity, totalPrice, uniqueParts };
  }, [items]);

  // Handlers
  const handleSort = (field: SortField) => {
    if (field === sortField) {
      setSortDirection((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDirection("asc");
    }
  };

  const handleSaveItem = useCallback(
    (item: BOMItem) => {
      setItems((prev) => {
        const existingIndex = prev.findIndex((i) => i.id === item.id);
        if (existingIndex >= 0) {
          const updated = [...prev];
          updated[existingIndex] = item;
          onChange?.(updated);
          return updated;
        }
        const updated = [...prev, item];
        onChange?.(updated);
        return updated;
      });
      setEditingItem(null);
      setIsAddingNew(false);
    },
    [onChange]
  );

  const handleDeleteSelected = useCallback(() => {
    if (selectedIds.size === 0) return;
    if (!confirm(`Delete ${selectedIds.size} selected item(s)?`)) return;

    setItems((prev) => {
      const updated = prev.filter((item) => !selectedIds.has(item.id));
      onChange?.(updated);
      return updated;
    });
    setSelectedIds(new Set());
  }, [selectedIds, onChange]);

  const handleExportCSV = useCallback(() => {
    const headers = [
      "Reference",
      "Part Number",
      "Manufacturer",
      "Description",
      "Quantity",
      "Footprint",
      "Value",
      "Price",
      "Supplier",
    ];
    const rows = items.map((item) => [
      item.reference,
      item.partNumber,
      item.manufacturer,
      item.description,
      item.quantity,
      item.footprint,
      item.value || "",
      item.price || "",
      item.supplier || "",
    ]);

    const csv = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `bom-${projectId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [items, projectId]);

  const handleImportCSV = useCallback(() => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".csv";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;

      const text = await file.text();
      const lines = text.split("\n").filter((l) => l.trim());
      const headers = lines[0].split(",").map((h) => h.trim().toLowerCase());

      const newItems: BOMItem[] = lines.slice(1).map((line) => {
        const values = line.split(",").map((v) => v.trim());
        return {
          id: crypto.randomUUID(),
          reference: values[headers.indexOf("reference")] || "",
          partNumber: values[headers.indexOf("part number")] || values[headers.indexOf("partnumber")] || "",
          manufacturer: values[headers.indexOf("manufacturer")] || "",
          description: values[headers.indexOf("description")] || "",
          quantity: parseInt(values[headers.indexOf("quantity")]) || 1,
          footprint: values[headers.indexOf("footprint")] || "",
          value: values[headers.indexOf("value")] || "",
          price: parseFloat(values[headers.indexOf("price")]) || undefined,
          supplier: values[headers.indexOf("supplier")] || "",
        };
      });

      setItems((prev) => {
        const updated = [...prev, ...newItems];
        onChange?.(updated);
        return updated;
      });
    };
    input.click();
  }, [onChange]);

  const handleSave = async () => {
    if (!onSave) return;
    setIsSaving(true);
    try {
      await onSave(items);
    } finally {
      setIsSaving(false);
    }
  };

  const SortIcon = ({ field }: { field: SortField }) => {
    if (field !== sortField) return null;
    return sortDirection === "asc" ? (
      <ChevronUp className="w-3 h-3 inline" />
    ) : (
      <ChevronDown className="w-3 h-3 inline" />
    );
  };

  return (
    <div className={cn("flex flex-col h-full bg-surface-primary rounded-lg border border-slate-700", className)}>
      {/* Toolbar */}
      <div className="flex items-center gap-2 p-3 border-b border-slate-700">
        <button
          onClick={() => setIsAddingNew(true)}
          className="flex items-center gap-1 px-3 py-1.5 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors text-sm"
        >
          <Plus className="w-4 h-4" />
          Add
        </button>

        <button
          onClick={handleDeleteSelected}
          disabled={selectedIds.size === 0}
          className="flex items-center gap-1 px-3 py-1.5 bg-red-600/20 text-red-400 rounded-lg hover:bg-red-600/30 transition-colors text-sm disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <Trash2 className="w-4 h-4" />
          Delete ({selectedIds.size})
        </button>

        <div className="flex-1" />

        <button
          onClick={handleImportCSV}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors text-sm"
        >
          <Upload className="w-4 h-4" />
          Import
        </button>

        <button
          onClick={handleExportCSV}
          className="flex items-center gap-1 px-3 py-1.5 bg-slate-700 text-slate-300 rounded-lg hover:bg-slate-600 transition-colors text-sm"
        >
          <Download className="w-4 h-4" />
          Export
        </button>

        {onSave && (
          <button
            onClick={handleSave}
            disabled={isSaving}
            className="flex items-center gap-1 px-3 py-1.5 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm disabled:opacity-50"
          >
            <Save className="w-4 h-4" />
            {isSaving ? "Saving..." : "Save"}
          </button>
        )}
      </div>

      {/* Search */}
      <div className="p-3 border-b border-slate-700">
        <div className="relative">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search parts..."
            className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:border-primary-500 focus:outline-none"
          />
        </div>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full text-sm">
          <thead className="bg-slate-800 sticky top-0">
            <tr>
              <th className="p-2 w-10">
                <input
                  type="checkbox"
                  checked={selectedIds.size === items.length && items.length > 0}
                  onChange={(e) => {
                    if (e.target.checked) {
                      setSelectedIds(new Set(items.map((i) => i.id)));
                    } else {
                      setSelectedIds(new Set());
                    }
                  }}
                  className="rounded border-slate-600"
                />
              </th>
              {[
                { field: "reference" as SortField, label: "Ref" },
                { field: "partNumber" as SortField, label: "Part Number" },
                { field: "manufacturer" as SortField, label: "Manufacturer" },
                { field: "description" as SortField, label: "Description" },
                { field: "quantity" as SortField, label: "Qty" },
                { field: "footprint" as SortField, label: "Footprint" },
                { field: "value" as SortField, label: "Value" },
              ].map(({ field, label }) => (
                <th
                  key={field}
                  onClick={() => handleSort(field)}
                  className="p-2 text-left text-slate-400 font-medium cursor-pointer hover:text-white"
                >
                  {label} <SortIcon field={field} />
                </th>
              ))}
              <th className="p-2 w-20">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filteredItems.map((item) => (
              <tr
                key={item.id}
                className={cn(
                  "border-b border-slate-700/50 hover:bg-slate-800/50",
                  selectedIds.has(item.id) && "bg-primary-600/10"
                )}
              >
                <td className="p-2">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(item.id)}
                    onChange={(e) => {
                      setSelectedIds((prev) => {
                        const next = new Set(prev);
                        if (e.target.checked) {
                          next.add(item.id);
                        } else {
                          next.delete(item.id);
                        }
                        return next;
                      });
                    }}
                    className="rounded border-slate-600"
                  />
                </td>
                <td className="p-2 text-cyan-400 font-mono">{item.reference}</td>
                <td className="p-2 text-white">{item.partNumber}</td>
                <td className="p-2 text-slate-400">{item.manufacturer}</td>
                <td className="p-2 text-slate-300 max-w-[200px] truncate">{item.description}</td>
                <td className="p-2 text-white text-center">{item.quantity}</td>
                <td className="p-2 text-slate-400 font-mono text-xs">{item.footprint}</td>
                <td className="p-2 text-yellow-400">{item.value}</td>
                <td className="p-2">
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => setEditingItem(item)}
                      className="p-1 hover:bg-slate-700 rounded"
                      title="Edit"
                    >
                      <Edit2 className="w-4 h-4 text-slate-400" />
                    </button>
                    {item.datasheetUrl && (
                      <a
                        href={item.datasheetUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="p-1 hover:bg-slate-700 rounded"
                        title="Datasheet"
                      >
                        <ExternalLink className="w-4 h-4 text-slate-400" />
                      </a>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>

        {filteredItems.length === 0 && (
          <div className="flex flex-col items-center justify-center py-12 text-slate-500">
            <Package className="w-12 h-12 mb-3" />
            <p>{searchQuery ? "No matching components" : "No components in BOM"}</p>
            {!searchQuery && (
              <button
                onClick={() => setIsAddingNew(true)}
                className="mt-4 text-primary-400 hover:text-primary-300"
              >
                Add your first component
              </button>
            )}
          </div>
        )}
      </div>

      {/* Footer with totals */}
      <div className="flex items-center gap-6 p-3 border-t border-slate-700 text-sm text-slate-400">
        <span>
          <strong className="text-white">{items.length}</strong> components
        </span>
        <span>
          <strong className="text-white">{totals.uniqueParts}</strong> unique parts
        </span>
        <span>
          <strong className="text-white">{totals.totalQuantity}</strong> total qty
        </span>
        {totals.totalPrice > 0 && (
          <span>
            Total: <strong className="text-green-400">${totals.totalPrice.toFixed(2)}</strong>
          </span>
        )}
      </div>

      {/* Edit Modal */}
      {(editingItem || isAddingNew) && (
        <EditModal
          item={editingItem}
          isNew={isAddingNew}
          onSave={handleSaveItem}
          onCancel={() => {
            setEditingItem(null);
            setIsAddingNew(false);
          }}
        />
      )}
    </div>
  );
}

export default BOMEditor;
