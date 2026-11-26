<script setup lang="ts">
import { FileDiff } from "lucide-vue-next";
import type {
  CodeDiffPayload,
  DiffFileAction,
  DiffFileBlock,
} from "../utils/codeDiff";

defineProps<{
  payload: CodeDiffPayload;
  compact?: boolean;
}>();

/** 将补丁操作类型映射为简短中文徽标文案。 */
function actionLabel(action: DiffFileAction): string {
  switch (action) {
    case "add":
      return "新增";
    case "delete":
      return "删除";
    case "move":
      return "移动";
    default:
      return "修改";
  }
}

/** 判断文件块内是否包含可用于展示的行号信息。 */
function hasLineNumbers(file: DiffFileBlock): boolean {
  return file.lines.some(
    (line) => line.oldNum !== undefined || line.newNum !== undefined,
  );
}
</script>

<template>
  <div :class="['code-diff-view', { compact }]">
    <div v-if="payload.files.length > 1" class="diff-summary-meta">
      <span class="diff-meta-chip">
        <FileDiff :size="11" />{{ payload.files.length }} 个文件
      </span>
      <span v-if="payload.totalAdditions" class="diff-stat-add">
        +{{ payload.totalAdditions }}
      </span>
      <span v-if="payload.totalDeletions" class="diff-stat-del">
        -{{ payload.totalDeletions }}
      </span>
    </div>

    <div
      v-for="(file, fileIdx) in payload.files"
      :key="file.path + ':' + fileIdx"
      class="diff-file-block"
    >
      <div class="diff-file-bar">
        <span class="diff-file-title">
          <span :class="['diff-action-badge', file.action]">
            {{ actionLabel(file.action) }}
          </span>
          <span class="diff-file-path">
            {{ file.path }}
            <template v-if="file.destination"> → {{ file.destination }}</template>
          </span>
          <span v-if="file.replaceAll" class="diff-flag-chip">全部替换</span>
        </span>
        <span class="diff-file-stats">
          <span v-if="file.additions" class="diff-stat-add">
            +{{ file.additions }}
          </span>
          <span v-if="file.deletions" class="diff-stat-del">
            -{{ file.deletions }}
          </span>
        </span>
      </div>

      <div
        v-if="file.action === 'delete' && !file.lines.length"
        class="diff-empty-note"
      >
        删除文件
      </div>
      <pre v-else class="diff-code"><code><span
        v-for="(line, lineIdx) in file.lines"
        :key="lineIdx"
        :class="['diff-line', 'diff-line-' + line.kind]"
      ><template v-if="line.kind === 'hunk'"><span class="diff-hunk-text">{{ line.text }}</span></template><template v-else><span
          v-if="hasLineNumbers(file)"
          class="diff-ln"
        >{{ line.oldNum ?? "" }}</span><span
          v-if="hasLineNumbers(file)"
          class="diff-ln"
        >{{ line.newNum ?? "" }}</span><span class="diff-sign">{{
          line.kind === "add" ? "+" : line.kind === "del" ? "-" : " "
        }}</span><span class="diff-line-text"><template v-if="line.text">{{
          line.text
        }}</template><template v-else>&#8203;</template></span></template></span></code></pre>
    </div>
  </div>
</template>
