package tfjvm;

import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Session;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.types.TFloat32;


public class Stress {
    public static void main(String[] args) {
    	Graph graph = new Graph(); 
        try (Session session = new Session(graph)) {
        	Ops ops = Ops.create(graph);
        	Shape shape = Shape.of(1000, 1000);
        	Operand<TFloat32> mat1 = ops.random.randomUniform(ops.constant(new long[] {10000, 10000}), TFloat32.class);
        	session.run(mat1);
        	Operand<TFloat32> mat2 = ops.random.randomUniform(ops.constant(new long[] {10000, 10000}), TFloat32.class);
        	session.run(mat2);
        	
        	Operand<TFloat32> mul = ops.linalg.matMul(mat1, mat2);
        	while(true) {
        		long start = System.currentTimeMillis();
        		session.run(mul);
        		long end = System.currentTimeMillis();
        		System.out.println((end - start) + "ms");
        	}
        }
    }
}
