"""
Script para comparar resultados entre diferentes variantes de ConvNeXtPose

Este script analiza y compara los resultados de testing de diferentes
variantes del modelo (XS, S, M, L) en Human3.6M.

Uso:
    python compare_variants.py
    python compare_variants.py --variants M L --epoch 70
    python compare_variants.py --protocol 2 --plot

Autor: Comparación de modelos ConvNeXtPose
Fecha: Octubre 2025
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Backend no-GUI
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("⚠️  Matplotlib no disponible. Las gráficas estarán deshabilitadas.")

from config_variants import MODEL_CONFIGS


def parse_args():
    """Parse argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Compare ConvNeXtPose variant results on Human3.6M'
    )
    
    parser.add_argument('--variants', nargs='+', default=['S', 'M', 'L'],
                       choices=['XS', 'S', 'M', 'L'],
                       help='Variants to compare (default: S M L)')
    parser.add_argument('--epoch', type=int, default=70,
                       help='Epoch number to compare (default: 70)')
    parser.add_argument('--protocol', type=int, default=2,
                       choices=[1, 2],
                       help='Protocol to analyze (default: 2)')
    parser.add_argument('--result_dir', type=str, default='../output/result',
                       help='Directory containing result files')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--save_report', action='store_true',
                       help='Save comparison report as markdown')
    
    return parser.parse_args()


def load_result_from_json(result_dir, variant, epoch):
    """
    Carga resultados desde archivo JSON
    
    Args:
        result_dir (str): Directorio de resultados
        variant (str): Variante del modelo
        epoch (int): Número de epoch
    
    Returns:
        dict: Resultados si se encuentra el archivo, None si no
    """
    # Intentar varios patrones de nombre
    patterns = [
        f"results_{variant}_epoch{epoch}.json",
        f"result_{variant}_epoch{epoch}.json",
        f"{variant}_epoch{epoch}_results.json",
    ]
    
    for pattern in patterns:
        filepath = os.path.join(result_dir, pattern)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    
    return None


def parse_eval_result_string(eval_string):
    """
    Parsea el string de evaluación para extraer métricas
    
    Args:
        eval_string (str): String con resultados de evaluación
    
    Returns:
        dict: Diccionario con métricas extraídas
    """
    results = {
        'total_error': None,
        'actions': {}
    }
    
    # Extraer error total
    if 'tot:' in eval_string:
        try:
            total_str = eval_string.split('tot:')[1].split('\n')[0].strip()
            results['total_error'] = float(total_str)
        except:
            pass
    
    # Extraer errores por acción
    action_names = [
        'Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning',
        'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking',
        'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether'
    ]
    
    for action in action_names:
        if action in eval_string:
            try:
                action_str = eval_string.split(action)[1].split(':')[1].split()[0]
                results['actions'][action] = float(action_str)
            except:
                pass
    
    return results


class VariantComparator:
    """
    Clase para comparar resultados entre variantes
    """
    
    def __init__(self, variants, epoch, protocol, result_dir):
        """
        Inicializa el comparador
        
        Args:
            variants (list): Lista de variantes a comparar
            epoch (int): Epoch a analizar
            protocol (int): Protocolo usado (1 o 2)
            result_dir (str): Directorio con resultados
        """
        self.variants = variants
        self.epoch = epoch
        self.protocol = protocol
        self.result_dir = result_dir
        self.results = {}
        
        # Cargar resultados
        self.load_all_results()
    
    def load_all_results(self):
        """Carga resultados de todas las variantes"""
        print("\n" + "="*80)
        print("Cargando resultados...")
        print("="*80 + "\n")
        
        for variant in self.variants:
            print(f"Buscando resultados para {variant}...", end=" ")
            result = load_result_from_json(self.result_dir, variant, self.epoch)
            
            if result:
                self.results[variant] = result
                print("✓ Encontrado")
            else:
                print("✗ No encontrado")
                print(f"  Intentado en: {self.result_dir}")
        
        if not self.results:
            print("\n❌ No se encontraron resultados para ninguna variante")
            print("Verifica que hayas ejecutado el testing primero con:")
            print("  python test_variants.py --variant M --gpu 0 --epoch 70")
            sys.exit(1)
        
        print(f"\n✓ Cargados resultados de {len(self.results)} variante(s)")
    
    def print_summary_table(self):
        """Imprime tabla resumen comparativa"""
        metric_name = 'PA-MPJPE' if self.protocol == 1 else 'MPJPE'
        
        print("\n" + "="*100)
        print(f"COMPARACIÓN DE RESULTADOS - HUMAN3.6M PROTOCOL {self.protocol} ({metric_name})")
        print("="*100)
        
        # Encabezado
        print(f"\n{'Variante':<12} {'Params (M)':<12} {'GFLOPs':<10} "
              f"{'Esperado (mm)':<15} {'Obtenido (mm)':<15} {'Diferencia':<15} {'Mejora vs S':<12}")
        print("-"*100)
        
        # Baseline (variante S si existe)
        baseline_result = None
        if 'S' in self.results:
            eval_result = self.results['S'].get('evaluation_result', '')
            parsed = parse_eval_result_string(eval_result)
            baseline_result = parsed.get('total_error')
        
        # Imprimir cada variante
        for variant in sorted(self.results.keys()):
            result = self.results[variant]
            config = MODEL_CONFIGS[variant]
            
            # Extraer métricas
            eval_result = result.get('evaluation_result', '')
            parsed = parse_eval_result_string(eval_result)
            obtained = parsed.get('total_error')
            
            # Calcular diferencia con esperado
            if self.protocol == 1:
                expected = config['expected_pa_mpjpe']
            else:
                expected = config['expected_mpjpe']
            
            difference = obtained - expected if obtained else None
            
            # Calcular mejora vs S
            improvement = ""
            if baseline_result and obtained and variant != 'S':
                imp_pct = ((baseline_result - obtained) / baseline_result) * 100
                improvement = f"{imp_pct:+.1f}%"
            elif variant == 'S':
                improvement = "baseline"
            
            # Formato de diferencia
            diff_str = f"{difference:+.2f} mm" if difference is not None else "N/A"
            if difference is not None:
                if abs(difference) <= 1.0:
                    diff_str += " ✓"
                elif difference > 0:
                    diff_str += " ⚠"
            
            # Imprimir fila
            obtained_str = f"{obtained:.2f}" if obtained else "N/A"
            print(f"{variant:<12} {config['params']:<12.1f} {config['gflops']:<10.1f} "
                  f"{expected:<15.1f} {obtained_str:<15} {diff_str:<15} {improvement:<12}")
        
        print("="*100)
        print("\nNotas:")
        print("  ✓ = Diferencia ≤ 1.0mm (excelente)")
        print("  ⚠ = Resultado peor que esperado")
        print("  Mejora vs S = Reducción porcentual de error respecto a variante S")
        print("="*100 + "\n")
    
    def print_action_comparison(self, variant):
        """Imprime comparación por acción para una variante"""
        if variant not in self.results:
            print(f"⚠️  No hay resultados para {variant}")
            return
        
        result = self.results[variant]
        eval_result = result.get('evaluation_result', '')
        parsed = parse_eval_result_string(eval_result)
        
        if not parsed['actions']:
            print(f"⚠️  No se encontraron resultados por acción para {variant}")
            return
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS POR ACCIÓN - {variant}")
        print(f"{'='*60}")
        print(f"\n{'Acción':<20} {'Error (mm)':<15}")
        print("-"*60)
        
        # Ordenar por error
        sorted_actions = sorted(parsed['actions'].items(), key=lambda x: x[1])
        
        for action, error in sorted_actions:
            print(f"{action:<20} {error:<15.2f}")
        
        if parsed['total_error']:
            print("-"*60)
            print(f"{'PROMEDIO':<20} {parsed['total_error']:<15.2f}")
        print("="*60 + "\n")
    
    def generate_comparison_plot(self, output_path='comparison_plot.png'):
        """Genera gráfico comparativo"""
        if not PLOTTING_AVAILABLE:
            print("⚠️  Matplotlib no disponible, saltando generación de gráficos")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Error total
        variants_list = sorted(self.results.keys())
        expected_errors = []
        obtained_errors = []
        params = []
        
        metric_name = 'PA-MPJPE' if self.protocol == 1 else 'MPJPE'
        
        for variant in variants_list:
            config = MODEL_CONFIGS[variant]
            result = self.results[variant]
            
            if self.protocol == 1:
                expected_errors.append(config['expected_pa_mpjpe'])
            else:
                expected_errors.append(config['expected_mpjpe'])
            
            eval_result = result.get('evaluation_result', '')
            parsed = parse_eval_result_string(eval_result)
            obtained = parsed.get('total_error', 0)
            obtained_errors.append(obtained)
            
            params.append(config['params'])
        
        # Gráfico de barras
        x = np.arange(len(variants_list))
        width = 0.35
        
        axes[0].bar(x - width/2, expected_errors, width, label='Esperado (paper)', alpha=0.8)
        axes[0].bar(x + width/2, obtained_errors, width, label='Obtenido', alpha=0.8)
        axes[0].set_xlabel('Variante')
        axes[0].set_ylabel(f'{metric_name} (mm)')
        axes[0].set_title(f'Comparación de {metric_name} - Protocol {self.protocol}')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(variants_list)
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Plot 2: Trade-off precisión vs complejidad
        axes[1].scatter(params, obtained_errors, s=200, alpha=0.6)
        for i, variant in enumerate(variants_list):
            axes[1].annotate(variant, (params[i], obtained_errors[i]),
                           xytext=(5, 5), textcoords='offset points')
        axes[1].set_xlabel('Parámetros (M)')
        axes[1].set_ylabel(f'{metric_name} (mm)')
        axes[1].set_title('Trade-off: Complejidad vs Precisión')
        axes[1].grid(True, alpha=0.3)
        axes[1].invert_yaxis()  # Menor error = mejor
        
        plt.tight_layout()
        
        # Guardar
        output_file = os.path.join(self.result_dir, output_path)
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Gráfico guardado en: {output_file}")
        plt.close()
    
    def generate_markdown_report(self, output_path='comparison_report.md'):
        """Genera reporte en formato markdown"""
        metric_name = 'PA-MPJPE' if self.protocol == 1 else 'MPJPE'
        
        report = f"""# Reporte de Comparación - ConvNeXtPose Variants

**Fecha**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Epoch**: {self.epoch}  
**Protocol**: {self.protocol} ({metric_name})  
**Dataset**: Human3.6M  

---

## Resumen de Resultados

| Variante | Params (M) | GFLOPs | Esperado (mm) | Obtenido (mm) | Diferencia | Estado |
|----------|------------|--------|---------------|---------------|------------|--------|
"""
        
        for variant in sorted(self.results.keys()):
            config = MODEL_CONFIGS[variant]
            result = self.results[variant]
            
            eval_result = result.get('evaluation_result', '')
            parsed = parse_eval_result_string(eval_result)
            obtained = parsed.get('total_error')
            
            if self.protocol == 1:
                expected = config['expected_pa_mpjpe']
            else:
                expected = config['expected_mpjpe']
            
            difference = obtained - expected if obtained else None
            
            status = ""
            if difference is not None:
                if abs(difference) <= 1.0:
                    status = "✅ Excelente"
                elif difference > 0 and difference <= 2.0:
                    status = "⚠️ Aceptable"
                elif difference > 2.0:
                    status = "❌ Revisar"
                else:
                    status = "🎯 Mejor que paper"
            
            obtained_str = f"{obtained:.2f}" if obtained else "N/A"
            diff_str = f"{difference:+.2f}" if difference is not None else "N/A"
            
            report += f"| {variant} | {config['params']:.1f} | {config['gflops']:.1f} | "
            report += f"{expected:.1f} | {obtained_str} | {diff_str} | {status} |\n"
        
        report += "\n---\n\n## Análisis Detallado\n\n"
        
        # Agregar análisis por variante
        for variant in sorted(self.results.keys()):
            report += f"### ConvNeXtPose-{variant}\n\n"
            
            result = self.results[variant]
            eval_result = result.get('evaluation_result', '')
            parsed = parse_eval_result_string(eval_result)
            
            if parsed['actions']:
                report += "#### Resultados por Acción\n\n"
                report += "| Acción | Error (mm) |\n|--------|------------|\n"
                
                for action, error in sorted(parsed['actions'].items()):
                    report += f"| {action} | {error:.2f} |\n"
                
                report += "\n"
        
        report += """---

## Interpretación de Resultados

- **✅ Excelente**: Diferencia ≤ 1.0mm con respecto al paper
- **⚠️ Aceptable**: Diferencia entre 1.0mm y 2.0mm
- **❌ Revisar**: Diferencia > 2.0mm (puede indicar problemas de configuración)
- **🎯 Mejor que paper**: Resultado superior al reportado

---

*Generado automáticamente por compare_variants.py*
"""
        
        # Guardar
        output_file = os.path.join(self.result_dir, output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📄 Reporte guardado en: {output_file}")


def main():
    """Función principal"""
    args = parse_args()
    
    # Crear comparador
    comparator = VariantComparator(
        variants=args.variants,
        epoch=args.epoch,
        protocol=args.protocol,
        result_dir=args.result_dir
    )
    
    # Mostrar tabla resumen
    comparator.print_summary_table()
    
    # Mostrar detalles por variante
    for variant in args.variants:
        if variant in comparator.results:
            comparator.print_action_comparison(variant)
    
    # Generar gráficos si se solicita
    if args.plot:
        print("\nGenerando gráficos comparativos...")
        comparator.generate_comparison_plot()
    
    # Generar reporte si se solicita
    if args.save_report:
        print("\nGenerando reporte markdown...")
        comparator.generate_markdown_report()
    
    print("\n✅ Comparación completada\n")


if __name__ == "__main__":
    main()
